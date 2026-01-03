
import dtlz_lib
import misda
import pandas as pd
import warnings
import warnings
warnings.filterwarnings("ignore")

"""
run_summary_test.py

A quick-check script to run a DTLZ5 analysis and print the full summary report.
Used to verify that the `MISDAResult.summary()` method runs without errors
and produces the expected output format.
"""

print("=== FULL SUMMARY for DTLZ5 ===")
Y5, _ = dtlz_lib.generate_dtlz5(N=300, M=3)
res5 = misda.analyze(Y5, caution=0.5, run_ses=True)
print(res5.summary())
