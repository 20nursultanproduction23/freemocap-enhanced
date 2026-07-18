"""
CLI for FreeMoCap Enhanced Post-Processing Pipeline

Usage:
    python -m freemocap_enhanced --recording-folder /path/to/recording
    python -m freemocap_enhanced --npy-file /path/to/skeleton.npy
"""
import argparse
import os
import sys
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="FreeMoCap Enhanced Post-Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process a FreeMoCap recording folder:
    python -m freemocap_enhanced --recording-folder "C:/data/my_recording"
  
  Process a single .npy file:
    python -m freemocap_enhanced --npy-file "skeleton_3d.npy"
  
  Custom parameters:
    python -m freemocap_enhanced --recording-folder ./rec --fps 60 --euro-cutoff 1.5 --no-bones
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--recording-folder", type=str,
        help="Path to FreeMoCap recording folder"
    )
    input_group.add_argument(
        "--npy-file", type=str,
        help="Path to individual .npy skeleton file (numFrames, numTrackedPoints, 3)"
    )
    
    parser.add_argument("--fps", type=float, default=30.0, help="Video frame rate (default: 30)")
    parser.add_argument("--output-suffix", type=str, default="_enhanced", help="Suffix for output files")
    parser.add_argument("--no-save", action="store_true", help="Don't save output to disk")
    
    filter_group = parser.add_argument_group("Filter Parameters")
    filter_group.add_argument("--euro-cutoff", type=float, default=1.0, help="One Euro min cutoff (default: 1.0, lower=more smoothing)")
    filter_group.add_argument("--euro-beta", type=float, default=0.007, help="One Euro beta (default: 0.007)")
    filter_group.add_argument("--butter-cutoff", type=float, default=6.0, help="Butterworth cutoff Hz (default: 6.0)")
    filter_group.add_argument("--butter-order", type=int, default=4, help="Butterworth order (default: 4)")
    
    constraint_group = parser.add_argument_group("Constraint Parameters")
    constraint_group.add_argument("--max-gap", type=int, default=10, help="Max NaN gap to interpolate (default: 10 frames)")
    constraint_group.add_argument("--outlier-threshold", type=float, default=5.0, help="Outlier detection threshold (default: 5.0)")
    constraint_group.add_argument("--bone-iterations", type=int, default=10, help="IK bone constraint iterations (default: 10)")
    constraint_group.add_argument("--no-bones", action="store_true", help="Skip bone length enforcement")
    constraint_group.add_argument("--no-foot-fix", action="store_true", help="Skip foot sliding fix")
    filter_group.add_argument("--foot-threshold", type=float, default=None, help="Foot contact velocity threshold (None=auto)")
    
    args = parser.parse_args()
    
    from pipeline import process_freemocap_recording, run_full_pipeline
    
    if args.npy_file:
        print(f"Loading .npy file: {args.npy_file}")
        skeleton_data = np.load(args.npy_file)
        print(f"Loaded: shape={skeleton_data.shape}")
        
        processed_data, report = run_full_pipeline(
            skeleton_data,
            fps=args.fps,
            one_euro_min_cutoff=args.euro_cutoff,
            one_euro_beta=args.euro_beta,
            butterworth_cutoff=args.butter_cutoff,
            butterworth_order=args.butter_order,
            max_gap=args.max_gap,
            outlier_threshold=args.outlier_threshold,
            enforce_bones=not args.no_bones,
            fix_foot_sliding=not args.no_foot_fix,
            bone_iterations=args.bone_iterations,
            velocity_threshold=args.foot_threshold,
        )
        
        if not args.no_save:
            output_path = args.npy_file.replace(".npy", f"{args.output_suffix}.npy")
            np.save(output_path, processed_data)
            print(f"\nSaved: {output_path}")
        
    elif args.recording_folder:
        processed_data, report = process_freemocap_recording(
            args.recording_folder,
            output_suffix=args.output_suffix,
            save_output=not args.no_save,
            fps=args.fps,
            one_euro_min_cutoff=args.euro_cutoff,
            one_euro_beta=args.euro_beta,
            butterworth_cutoff=args.butter_cutoff,
            butterworth_order=args.butter_order,
            max_gap=args.max_gap,
            outlier_threshold=args.outlier_threshold,
            enforce_bones=not args.no_bones,
            fix_foot_sliding=not args.no_foot_fix,
            bone_iterations=args.bone_iterations,
            velocity_threshold=args.foot_threshold,
        )
    
    report_path = os.path.join(
        os.path.dirname(args.npy_file or args.recording_folder),
        f"enhanced_pipeline_report{args.output_suffix}.json"
    )
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
