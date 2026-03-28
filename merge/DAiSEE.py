import shutil
import subprocess
import sys
from pathlib import Path


def main():
    script_dir = Path(__file__).parent.resolve()
    daisee_dir = script_dir.parent / "datasets" / "DAiSEE"
    mediapipe_script = daisee_dir / "mediapipe-crop-and-labelv2.py"
    output_source = daisee_dir / "output"
    output_dest = script_dir / "output"

    if not mediapipe_script.exists():
        print(f"Error: {mediapipe_script} not found")
        sys.exit(1)

    print("=" * 50)
    print("Running mediapipe-crop-and-labelv2.py")
    print("=" * 50)

    try:
        result = subprocess.run(
            [sys.executable, str(mediapipe_script)],
            cwd=str(daisee_dir),
            check=True,
            capture_output=False,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running mediapipe script: {e}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("Moving results to merge/output")
    print("=" * 50)

    if not output_source.exists():
        print(f"Error: Output directory not found at {output_source}")
        sys.exit(1)

    output_dest.mkdir(parents=True, exist_ok=True)

    for item in output_source.iterdir():
        dest_item = output_dest / item.name

        if dest_item.exists():
            if dest_item.is_dir():
                shutil.rmtree(dest_item)
            else:
                dest_item.unlink()

        if item.is_dir():
            shutil.copytree(item, dest_item)
            print(f"Copied: {item.name}")
        else:
            shutil.copy2(item, dest_item)
            print(f"Copied: {item.name}")

    print(f"\nResults moved successfully to: {output_dest}")

    print("\nContents in merge/output:")
    for item in sorted(output_dest.iterdir()):
        if item.is_dir():
            print(f"  {item.name}/")
        else:
            print(f"  {item.name}")

    print("\n" + "=" * 50)
    print("Process completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
