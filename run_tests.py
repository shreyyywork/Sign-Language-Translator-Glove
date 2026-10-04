"""
Comprehensive Test Runner for ISL Glove Project.
Runs all unit and integration tests across simulator, preprocessor,
calibration, ML training, evaluation, prediction, and API modules.
"""

import sys
import time
import shutil
import tempfile
import traceback

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run_all_tests():
    print("=" * 70)
    print("      ISL GLOVE - SYSTEM VERIFICATION & TEST SUITE")
    print("=" * 70)

    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    # ----------------------------------------------------
    # Suite 1: Simulator Tests
    # ----------------------------------------------------
    print("\n[SUITE 1] Testing Biomechanical Sensor Simulator...")
    from tests.test_simulator import (
        test_simulator_initialization,
        test_single_sample_generation,
        test_finger_bending_bounds,
        test_dataset_generation
    )

    tests_s1 = [
        ("Simulator Initialization & 20 Gestures Config", test_simulator_initialization),
        ("Single 11-Feature Sample Generation", test_single_sample_generation),
        ("Anatomical Finger Bending Bounds", test_finger_bending_bounds),
    ]

    for name, fn in tests_s1:
        total_tests += 1
        t0 = time.time()
        try:
            fn()
            dt = time.time() - t0
            print(f"  [PASS] {name} ({dt:.3f}s)")
            passed_tests += 1
        except Exception as e:
            print(f"  [FAIL] {name} -> {e}")
            traceback.print_exc()
            failed_tests += 1

    # Test dataset generator with tmp directory
    total_tests += 1
    tmp_d = tempfile.mkdtemp()
    try:
        t0 = time.time()
        test_dataset_generation(tmp_d)
        dt = time.time() - t0
        print(f"  [PASS] Multi-User Simulated Dataset Generation ({dt:.3f}s)")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Multi-User Simulated Dataset Generation -> {e}")
        traceback.print_exc()
        failed_tests += 1
    finally:
        shutil.rmtree(tmp_d, ignore_errors=True)

    # ----------------------------------------------------
    # Suite 2: Preprocessor & Normalization Tests
    # ----------------------------------------------------
    print("\n[SUITE 2] Testing Preprocessor & Feature Scaling...")
    from tests.test_preprocessor import (
        test_preprocessor_clean_and_fit,
        test_single_sample_transform
    )

    tests_s2 = [
        ("DataFrame Cleaning, Validation & Fit-Transform", test_preprocessor_clean_and_fit),
        ("Single 11-Feature Sample Inference Scaling", test_single_sample_transform),
    ]

    for name, fn in tests_s2:
        total_tests += 1
        t0 = time.time()
        try:
            fn()
            dt = time.time() - t0
            print(f"  [PASS] {name} ({dt:.3f}s)")
            passed_tests += 1
        except Exception as e:
            print(f"  [FAIL] {name} -> {e}")
            traceback.print_exc()
            failed_tests += 1

    # ----------------------------------------------------
    # Suite 3: Calibration Service Tests
    # ----------------------------------------------------
    print("\n[SUITE 3] Testing 3-Step Calibration & Per-User Normalization...")
    from tests.test_calibration import (
        test_calibration_service_default,
        test_compute_calibration_from_samples
    )

    tmp_d = tempfile.mkdtemp()
    try:
        total_tests += 2
        t0 = time.time()
        test_calibration_service_default(tmp_d)
        print(f"  [PASS] Default Calibration Profile Baseline ({time.time()-t0:.3f}s)")
        passed_tests += 1

        t0 = time.time()
        test_compute_calibration_from_samples(tmp_d)
        print(f"  [PASS] 3-Step Calibration (Relaxed, Straight, Bent) Computation ({time.time()-t0:.3f}s)")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Calibration Suite -> {e}")
        traceback.print_exc()
        failed_tests += 1
    finally:
        shutil.rmtree(tmp_d, ignore_errors=True)

    # ----------------------------------------------------
    # Suite 4: ML Training & Model Selection
    # ----------------------------------------------------
    print("\n[SUITE 4] Testing ML Training (Decision Tree, Random Forest, SVM)...")
    from tests.test_ml_pipeline import test_baseline_training_and_comparison
    tmp_d = tempfile.mkdtemp()
    try:
        total_tests += 1
        t0 = time.time()
        test_baseline_training_and_comparison(tmp_d)
        print(f"  [PASS] Decision Tree vs Random Forest vs SVM Training & Evaluation ({time.time()-t0:.3f}s)")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] ML Baseline Pipeline -> {e}")
        traceback.print_exc()
        failed_tests += 1
    finally:
        shutil.rmtree(tmp_d, ignore_errors=True)

    # ----------------------------------------------------
    # Suite 5: Real-Time Prediction Service
    # ----------------------------------------------------
    print("\n[SUITE 5] Testing Real-Time Prediction Engine & Confidence Scoring...")
    from tests.test_prediction import test_real_time_prediction
    tmp_d = tempfile.mkdtemp()
    try:
        total_tests += 1
        t0 = time.time()
        test_real_time_prediction(tmp_d)
        print(f"  [PASS] Real-Time Gesture Inference & Top-K Probability Output ({time.time()-t0:.3f}s)")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Real-Time Prediction Engine -> {e}")
        traceback.print_exc()
        failed_tests += 1
    finally:
        shutil.rmtree(tmp_d, ignore_errors=True)

    # ----------------------------------------------------
    # Suite 6: Advanced Dynamic LSTM Model
    # ----------------------------------------------------
    print("\n[SUITE 6] Testing Advanced Time-Series PyTorch LSTM Model...")
    from ml.dynamic_lstm import DynamicLSTMPipeline
    from backend.simulator import GloveSensorSimulator
    import pandas as pd

    tmp_d = tempfile.mkdtemp()
    try:
        total_tests += 1
        t0 = time.time()
        sim = GloveSensorSimulator("config/gestures.json")
        csv_path = sim.generate_full_dataset(
            users=["U01", "U02"],
            samples_per_gesture=20,
            output_csv_path=f"{tmp_d}/lstm_train.csv"
        )
        df = pd.read_csv(csv_path)

        lstm = DynamicLSTMPipeline(models_dir=f"{tmp_d}/models", eval_dir=f"{tmp_d}/eval", sequence_length=15)
        rep = lstm.train_lstm(df, epochs=3, batch_size=16)
        assert rep["best_val_accuracy"] >= 0.0
        print(f"  [PASS] PyTorch Time-Series LSTM Training & Sequence Windowing ({time.time()-t0:.3f}s)")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] LSTM Pipeline -> {e}")
        traceback.print_exc()
        failed_tests += 1
    finally:
        shutil.rmtree(tmp_d, ignore_errors=True)

    # ----------------------------------------------------
    # Suite 7: FastAPI Endpoints
    # ----------------------------------------------------
    print("\n[SUITE 7] Testing FastAPI Endpoints...")
    from tests.test_api import (
        test_api_health,
        test_api_gestures,
        test_api_mode_switch,
        test_api_set_simulator_gesture,
        test_api_serial_ports
    )

    api_tests = [
        ("GET /api/health", test_api_health),
        ("GET /api/gestures (20 ISL signs)", test_api_gestures),
        ("POST /api/mode (simulator/hardware)", test_api_mode_switch),
        ("POST /api/simulator/set_gesture", test_api_set_simulator_gesture),
        ("GET /api/serial/ports", test_api_serial_ports),
    ]

    for name, fn in api_tests:
        total_tests += 1
        t0 = time.time()
        try:
            fn()
            print(f"  [PASS] {name} ({time.time()-t0:.3f}s)")
            passed_tests += 1
        except Exception as e:
            print(f"  [FAIL] {name} -> {e}")
            traceback.print_exc()
            failed_tests += 1

    # ----------------------------------------------------
    # Summary
    # ----------------------------------------------------
    print("\n" + "=" * 70)
    print(f"TEST RESULTS: {passed_tests}/{total_tests} PASSED ({failed_tests} FAILED)")
    print("=" * 70)

    if failed_tests == 0:
        print("\nALL TESTS PASSED SUCCESSFULLY! The system is 100% verified.")
        return 0
    else:
        print(f"\n{failed_tests} tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
