#!/usr/bin/env python
# coding: utf-8

"""
Main script to run all parts of Project 2
Parts: B, C, D, E, F
"""

import sys
from pathlib import Path

# Ensure the script can find the part modules
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

def run_all_parts():
    """Run all parts of the project in sequence"""
    
    print("=" * 80)
    print("PROJECT 2 - MAIN EXECUTION")
    print("=" * 80)
    
    # Part B
    print("\n" + "=" * 80)
    print("RUNNING PART B")
    print("=" * 80)
    try:
        from Project2_Part_b import main as run_part_b
        run_part_b()
        print("✓ Part B completed successfully")
    except Exception as e:
        print(f"✗ Part B failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Part C
    print("\n" + "=" * 80)
    print("RUNNING PART C")
    print("=" * 80)
    try:
        from Project2_Part_c import main as run_part_c
        run_part_c()
        print("✓ Part C completed successfully")
    except Exception as e:
        print(f"✗ Part C failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Part D
    print("\n" + "=" * 80)
    print("RUNNING PART D")
    print("=" * 80)
    try:
        from Project2_Part_d import main as run_part_d
        run_part_d()
        print("✓ Part D completed successfully")
    except Exception as e:
        print(f"✗ Part D failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Part E (if available)
    print("\n" + "=" * 80)
    print("RUNNING PART E")
    print("=" * 80)
    try:
        from Project2_Part_e import main as run_part_e
        run_part_e()
        print("✓ Part E completed successfully")
    except ModuleNotFoundError:
        print("ℹ Part E not found - skipping")
    except Exception as e:
        print(f"✗ Part E failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Part F
    print("\n" + "=" * 80)
    print("RUNNING PART F")
    print("=" * 80)
    try:
        from Project2_Part_f import main as run_part_f
        trained_layers, loss_history, accuracy_history, test_accuracy = run_part_f()
        print(f"✓ Part F completed successfully (Test Accuracy: {test_accuracy:.4f})")
    except Exception as e:
        print(f"✗ Part F failed: {e}")
        import traceback
        traceback.print_exc()
    

def run_specific_part(part):
    """Run a specific part only"""
    part = part.upper()
    
    print("=" * 80)
    print(f"RUNNING PART {part}")
    print("=" * 80)
    
    part_modules = {
        'B': 'Project2_Part_b',
        'C': 'Project2_Part_c',
        'D': 'Project2_Part_d',
        'E': 'Project2_Part_e',
        'F': 'Project2_Part_f',
        'G': 'Project2_Part_g'
    }
    
    if part not in part_modules:
        print(f"Error: Invalid part '{part}'. Valid parts are: B, C, D, E, F")
        return
    
    try:
        module = __import__(part_modules[part], fromlist=['main'])
        module.main()
        print(f"\n✓ Part {part} completed successfully")
    except ModuleNotFoundError:
        print(f"ℹ Part {part} not found - module {part_modules[part]}.py doesn't exist")
    except Exception as e:
        print(f"✗ Part {part} failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific part if provided as argument
        part = sys.argv[1]
        run_specific_part(part)
    else:
        # Run all parts
        run_all_parts()
