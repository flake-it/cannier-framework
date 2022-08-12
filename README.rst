=================
CANNIER-Framework
=================

CANNIER-Framework is command-line tool that automates the empirical evaluation for the `ESE <https://www.springer.com/journal/10664>`_ paper "Empirically Evaluating Flaky Test Detection Techniques Combining Test Case Rerunning and Machine Learning Models". Part of its function is to automatically use `pytest-CANNIER <https://github.com/flake-it/pytest-cannier>`_.

Prerequisites
=============

The dependencies of CANNIER-Framework can be found in ``requirements.txt``. It also requires ``git``, ``docker``, and ``virtualenv`` to be installed on the system. We have only tested CANNIER-Framework on Ubuntu 20.04 and Python 3.8. We cannot guarantee correct results with other environments.

Installation
============

You can install pytest-CANNIER with ``pip install PATH`` where ``PATH`` is the directory containing ``setup.py``. This will also install the dependencies.

Usage
=====

You can use CANNIER-Framework with ``cannier COMMAND *ARGS``. ``COMMAND`` can be one of:

- ``setup`` Setup the subject projects as part of the build stage for the ``canner-experiment`` Docker image (see the `CANNIER-Experiment <https://github.com/flake-it/cannier-expierment>`_ repository for more details). This command is not for manual use.
- ``manage`` Execute a project's test suite with pytest-CANNIER inside a ``canner-experiment`` Docker container. ``ARGS`` must provide the name of the project, the mode for pytest-CANNIER, a unique number to differentiate this container from other containers for the same project and mode, the name of the victim test case when the mode is ``victim`` (empty string otherwise), and the commands required to execute the test suite (typically ``python -m pytest``). This command is not for manual use.
- ``run`` Start containers to run CANNIER-Framework with the ``manage`` command for every project and the modes specified by ``ARGS``. The containers running ``churn`` must finish before those running ``features``. The containers running ``baseline`` and ``shuffle`` must finish before those running ``victim``. 
- ``collate`` Collate the outcome and feature data recorded by pytest-CANNIER.
- ``shap`` Train an extra trees model and apply the SHAP technique for each of the four flaky test classification problems described in the paper. Can only be used after ``collate``.
- ``preds`` Train and evaluate an extra trees model using stratified 10-fold cross validation for each classification problem and feature sample size from 1 to 15.  Can only be used after ``collate``.
- ``points`` Find the Pareto front of detection performance and time cost for the three rerunning-based flaky test detection techniques described in the paper. Can only be used after ``preds``.
- ``figures`` Generate the data for the tables and figures in the paper. Can only be used after ``points``.

CANNIER-Framework also offers the following options:

- ``--processes={PROCESSES}`` Maximum number of parallel processes to use (default is the result of calling ``os.cpu_count``).
- ``--timeout={TIMEOUT}`` Maximum run time for containers in seconds (default 28800).
- ``--n-repeats={N_REPEATS}`` Number of test suite runs with pytest-CANNIER for each project when the mode is ``features`` and the number of times to repeat model training and evaluation (default 30).
- ``--n-reruns={N_REPEATS}`` Number of test suite runs with pytest-CANNIER for each project when the mode is either ``baseline`` or ``shuffle`` (default 2,500).

Output
======

The output of CANNIER-Framework depends on ``COMMAND``:

- ``run`` A directory named ``volume`` with subdirectories for each subject project. These will contain an `SQLite <https://www.sqlite.org/index.html>`_ database with the results of pytest-CANNIER.
- ``collate`` This will produce three files:

    - ``items.npy`` A `NumPy <https://numpy.org/>`_ array with shape ``(N_TESTS, 7)``, where ``N_TESTS`` is the total number of test cases across all projects. From left-to-right, the columns indicate: which project the test case is from (an integer id), the number of times the test case failed in the ``baseline`` mode of pytest-CANNIER, the number of times the test case failed in the ``shuffle`` mode, if the test case is NOD flaky (0 = false, 1 = true), if the test case is a victim, and if the test case is relevant to the NOD-vs-Victim flaky test classification problem.
    - ``features.npy`` A NumPy array with shape ``(N_TESTS, N_REPEATS, 18)`` containing the ``N_REPEATS`` sets of the 18 test case features measured by pytest-CANNIER in the ``features`` mode.
    - ``dependencies.pkl`` A `pickle <https://docs.python.org/3/library/pickle.html>`_ file containing a list of boolean NumPy arrays for each project. The arrays are packed with ``numpy.packbits`` and can be unpacked with ``numpy.unpackbits``. Once an array is unpacked, its shape is ``(N_TESTS_PROJ, N_TESTS_PROJ)``, where ``N_TESTS_PROJ`` is the number of test cases in the project. The value at ``[i, j]`` indicates if test case ``j`` is a polluter of test case ``i``.
- ``shap`` A directory named ``shap`` containing the SHAP value matrix for each classification problem as a NumPy array with shape ``(N_TESTS, 18)``. 
- ``preds`` A directory named ``preds`` containing the predicted probabilities of the extra trees model for each classification problem and feature sample size from 1 to 15 as a NumPy array. The arrays are named ``{PROBLEM}_{N_SAMPLES}.npy``, where ``PROBLEM`` is the classification problem and ``N_SAMPLES`` is the sample size, with shape ``(N_TESTS, N_REPEATS)``. A given row contains the ``N_REPEATS`` predicted probabilities of the test case being in the positive class of ``PROBLEM``.
- ``points`` A directory named ``points`` containing the detection performance, time cost, and parameters of the points on the Pareto fronts for the three rerunning-based flaky test detection techniques as a NumPy array.
- ``figures`` Directories named ``tables`` and ``plots`` containing `LaTeX <https://docs.python.org/3/library/pickle.html>`_ code.

Testing
=======

CANNIER-Framework has its own `pytest <https://docs.pytest.org/en/7.1.x/>`_ test suite. To execute it, you must pass the ``--schema-file={SCHEMA_FILE}`` where ``SCHEMA_FILE`` is the path to the schema file for the database. This can be found in the CANNIER-Experiment repository.