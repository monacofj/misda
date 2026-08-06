
import mop_definitions
import misda
import warnings
warnings.filterwarnings("ignore")

"""
run_summary_test.py

A quick-check script to run a DTLZ5 analysis and print the full summary report.
Used to verify that the `MISDAResult.summary()` method runs without errors
and produces the expected output format.
"""

def main():
    print("=== FULL SUMMARY for DTLZ5 ===")
    Y5, _ = mop_definitions.generate_dtlz5(N=300, M=3)
    result = misda.analyze(Y5, aggressiveness=0.5)
    print(result.summary())


if __name__ == "__main__":
    main()
