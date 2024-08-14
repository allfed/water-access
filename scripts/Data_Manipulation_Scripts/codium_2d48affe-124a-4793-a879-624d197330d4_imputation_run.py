import argparse
import json
import sys

import pytest

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to file or folder to run")
    namespace = parser.parse_args()
    path = namespace.path
    file_name = path.split("/")[-1]
    original_file_name = file_name.replace("codium_", "")

    args = [path]

    if int(pytest.__version__.split('.')[0]) >= 6:
        args += ["--no-header", "--no-summary", "-vv", "--disable-warnings"]

    class ResultsCollector:
        def __init__(self):
            self.reports = {}

        @pytest.hookimpl(hookwrapper=True)
        def pytest_exception_interact(self, node, call, report):
            outcome = yield
            if report.when == 'collect':
                self.reports["codium_tests_results_error"] = str(report.longreprtext
                                                                 .replace(file_name, original_file_name))

        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_makereport(self):
            outcome = yield
            report = outcome.get_result()
            if report.when == 'call':
                self.reports[report.head_line.split(".")[-1]] = {"passed": report.passed,
                                                                 "message": report.longreprtext}


    results = "{}"
    try:
        test_results = ResultsCollector()
        pytest.main(args, [test_results])
        results = json.dumps(test_results.reports)
    except Exception as e:
        results = json.dumps({"codium_tests_results_error": str(e)})
    finally:
        if results is not None:
            print("=== Codium Tests Results ===")
            print(results)
            print("=== End Codium Tests Results ===")
        sys.exit(0)
