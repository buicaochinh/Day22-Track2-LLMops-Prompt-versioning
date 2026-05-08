
import argparse
import subprocess
import sys

def run_script(script_name):
    print(f"\n" + "="*60)
    print(f"🚀 Running: {script_name}")
    print("="*60 + "\n")
    
    try:
        # Using sys.executable to ensure we use the same python environment
        result = subprocess.run([sys.executable, script_name], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {script_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run all or specific steps of the Day 22 Lab.")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4], help="Run only a specific step (1-4)")
    args = parser.parse_args()

    steps = {
        1: "01_langsmith_rag_pipeline.py",
        2: "02_prompt_hub_ab_routing.py",
        3: "03_ragas_evaluation.py",
        4: "04_guardrails_validator.py"
    }

    if args.step:
        # Run only the specified step
        script = steps[args.step]
        success = run_script(script)
        if success:
            print(f"\n✅ Step {args.step} completed successfully.")
        else:
            sys.exit(1)
    else:
        # Run all steps sequentially
        print("🧪 Starting Day 22 Lab - Full Execution Pipeline")
        for step_num in sorted(steps.keys()):
            script = steps[step_num]
            if not run_script(script):
                print(f"\n🛑 Pipeline stopped at Step {step_num} due to errors.")
                sys.exit(1)
        
        print("\n" + "🎉" * 20)
        print("  ALL STEPS COMPLETED SUCCESSFULLY!")
        print("  Remember to check your evidence/ folder before submitting.")
        print("🎉" * 20)

if __name__ == "__main__":
    main()
