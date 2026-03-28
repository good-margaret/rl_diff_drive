# # scripts/record_gif_after_training.py
# """
# Record GIF from trained model without modifying training code.

# Supports:
#   - Custom PPO models (.pt)
#   - Stable-Baselines3 models (.zip)
#   - Custom SAC/TD3 models (.pt)

# Usage:
#   # Record from custom PPO model
#   python scripts/record_gif_after_training.py --model models/ppo/ppo_improved.pt --output demo.gif
  
#   # Record from SB3 model
#   python scripts/record_gif_after_training.py --model models/ppo/ppo_default.zip --output demo.gif
  
#   # Record with specific start position
#   python scripts/record_gif_after_training.py --model models/ppo/ppo_improved.pt --x 8.0 --y 8.0 --angle 0
  
#   # Record from multiple start positions
#   python scripts/record_gif_after_training.py --model models/ppo/ppo_improved.pt --multi --output-dir gifs/
  
#   # List available models
#   python scripts/record_gif_after_training.py --list-models
# """

# import os
# import sys
# import argparse
# import numpy as np
# from pathlib import Path

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from src.environment import DiffDriveEnv


# def find_models(model_dir="models"):
#     """Find all available models in the directory."""
#     models = []
#     model_path = Path(model_dir)
    
#     if model_path.exists():
#         # Find .pt files (custom models)
#         for file in model_path.glob("**/*.pt"):
#             models.append({
#                 'path': str(file),
#                 'name': file.stem,
#                 'type': 'Custom PPO/SAC/TD3',
#                 'algo': file.parent.name if file.parent.name != 'models' else 'unknown'
#             })
        
#         # Find .zip files (Stable-Baselines3 models)
#         for file in model_path.glob("**/*.zip"):
#             models.append({
#                 'path': str(file),
#                 'name': file.stem,
#                 'type': 'Stable-Baselines3',
#                 'algo': file.parent.name if file.parent.name != 'models' else 'unknown'
#             })
    
#     return models


# def load_model(model_path):
#     """Load model based on file extension."""
#     if model_path.endswith('.zip'):
#         return load_sb3_model(model_path)
#     elif model_path.endswith('.pt'):
#         return load_custom_model(model_path)
#     else:
#         print(f"Unsupported model format: {model_path}")
#         return None


# def load_sb3_model(model_path):
#     """Load Stable-Baselines3 model."""
#     try:
#         from stable_baselines3 import PPO, SAC, TD3
        
#         # Try different algorithms
#         try:
#             model = PPO.load(model_path)
#             print(f"Loaded SB3 PPO model: {model_path}")
#             return model, 'sb3'
#         except:
#             try:
#                 model = SAC.load(model_path)
#                 print(f"Loaded SB3 SAC model: {model_path}")
#                 return model, 'sb3'
#             except:
#                 try:
#                     model = TD3.load(model_path)
#                     print(f"Loaded SB3 TD3 model: {model_path}")
#                     return model, 'sb3'
#                 except Exception as e:
#                     print(f"Failed to load SB3 model: {e}")
#                     return None, None
#     except ImportError:
#         print("Error: stable-baselines3 not installed")
#         return None, None


# def load_custom_model(model_path):
#     """Load custom PyTorch model."""
#     try:
#         # Try to load as PPO model first
#         from models.ppo_model import PPOModel
#         try:
#             model = PPOModel.load(model_path)
#             print(f"Loaded custom PPO model: {model_path}")
#             return model, 'custom'
#         except:
#             pass
        
#         # Try other custom models
#         try:
#             from models.sac_model import SACModel
#             model = SACModel.load(model_path)
#             print(f"Loaded custom SAC model: {model_path}")
#             return model, 'custom'
#         except:
#             pass
        
#         try:
#             from models.td3_model import TD3Model
#             model = TD3Model.load(model_path)
#             print(f"Loaded custom TD3 model: {model_path}")
#             return model, 'custom'
#         except:
#             pass
        
#         print(f"Could not load custom model: {model_path}")
#         return None, None
        
#     except ImportError as e:
#         print(f"Error loading custom model: {e}")
#         return None, None


# def record_gif(model, model_type, output_path, fps=30, max_steps=600, 
#                start_position=None, deterministic=True):
#     """Record a GIF from a trained model."""
#     try:
#         import imageio
#     except ImportError:
#         print("Error: imageio not installed. Install with: pip install imageio imageio-ffmpeg")
#         return False
    
#     # Create environment
#     env = DiffDriveEnv(render_mode='rgb_array')
    
#     # Set start position
#     if start_position is not None:
#         env.x, env.y, env.theta = start_position
#         env.trajectory = [(env.x, env.y, env.theta)]
#         print(f"Recording from custom position: x={env.x:.2f}, y={env.y:.2f}, angle={np.degrees(env.theta):.1f}°")
#     else:
#         env.reset()
#         print(f"Recording from random start: x={env.x:.2f}, y={env.y:.2f}, angle={np.degrees(env.theta):.1f}°")
    
#     # Record frames
#     frames = []
    
#     # Get initial observation
#     if hasattr(env, '_obs'):
#         obs = env._obs()
#     else:
#         x, y, theta = env.get_state()
#         dist = np.hypot(x, y)
#         obs = np.array([
#             x / env.FIELD,
#             y / env.FIELD,
#             np.cos(theta),
#             np.sin(theta),
#             dist / (env.FIELD * np.sqrt(2)),
#         ], dtype=np.float32)
    
#     done = False
#     step = 0
    
#     print("Recording...", end="", flush=True)
    
#     while not done and step < max_steps:
#         # Get action from model
#         if model_type == 'sb3':
#             action, _ = model.predict(obs, deterministic=deterministic)
#         else:  # custom model
#             action = model.predict(obs, deterministic=deterministic)
        
#         # Step environment
#         obs, reward, terminated, truncated, info = env.step(action)
#         done = terminated or truncated
#         step += 1
        
#         # Capture frame
#         try:
#             frame = env.render(mode='rgb_array')
#             if frame is not None:
#                 frames.append(frame)
#         except Exception as e:
#             print(f"\nWarning: Could not render frame: {e}")
        
#         if step % 100 == 0:
#             print(".", end="", flush=True)
    
#     print(" done!")
    
#     # Save GIF
#     if frames:
#         # Create output directory if needed
#         os.makedirs(os.path.dirname(os.path.abspath(output_path)) if os.path.dirname(output_path) else '.', exist_ok=True)
        
#         try:
#             with imageio.get_writer(output_path, mode='I', fps=fps, loop=0) as writer:
#                 for frame in frames:
#                     writer.append_data(frame)
#             print(f"GIF saved: {output_path}")
#             print(f"  Frames: {len(frames)}")
#             print(f"  Duration: {len(frames)/fps:.1f} seconds")
#             print(f"  Steps: {step}")
#             print(f"  Final distance: {info['dist']:.3f} m")
#             print(f"  Success: {info.get('success', False)}")
#             return True
#         except Exception as e:
#             print(f"Error saving GIF: {e}")
#             return False
#     else:
#         print("Warning: No frames captured!")
#         return False
    
#     env.close()


# def record_multiple_gifs(model, model_type, output_dir, fps=30, max_steps=600):
#     """Record GIFs from multiple start positions."""
#     os.makedirs(output_dir, exist_ok=True)
    
#     start_positions = [
#         (None, "random"),
#         (8.0, 8.0, 0.0, "far_corner_forward"),
#         (8.0, 8.0, 2.0, "far_corner_wrong"),
#         (-8.0, -8.0, 0.0, "opposite_corner"),
#         (5.0, -5.0, 1.57, "side_facing_away"),
#         (0.0, 8.0, 3.14, "top_facing_down"),
#         (-5.0, 5.0, -1.57, "bottom_facing_up"),
#         (8.0, 0.0, 1.57, "right_facing_up"),
#         (-8.0, 0.0, -1.57, "left_facing_down"),
#     ]
    
#     successful = 0
#     for pos in start_positions:
#         if len(pos) == 4:
#             x, y, angle, name = pos
#             start_pos = (x, y, angle)
#             output_path = os.path.join(output_dir, f"{name}.gif")
#         else:
#             start_pos = None
#             output_path = os.path.join(output_dir, f"random_start.gif")
        
#         print(f"\n{'='*50}")
#         if record_gif(model, model_type, output_path, fps, max_steps, start_pos):
#             successful += 1
    
#     print(f"\n{'='*50}")
#     print(f"Recorded {successful}/{len(start_positions)} GIFs")
#     print(f"Saved to: {output_dir}")


# def list_available_models():
#     """List all available models."""
#     print("\nAvailable models:")
#     models = find_models()
    
#     if not models:
#         print("  No models found in models/ directory")
#         return
    
#     # Group by algorithm
#     models_by_algo = {}
#     for model in models:
#         algo = model['algo'].upper()
#         if algo not in models_by_algo:
#             models_by_algo[algo] = []
#         models_by_algo[algo].append(model)
    
#     for algo, models_list in models_by_algo.items():
#         print(f"\n{algo}:")
#         for model in models_list:
#             print(f"  • {model['name']} ({model['type']})")
#             print(f"    Path: {model['path']}")


# def main():
#     parser = argparse.ArgumentParser(
#         description="Record GIF from trained model",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   # Record from custom PPO model
#   python scripts/record_gif_after_training.py --model models/ppo/ppo_improved.pt --output demo.gif
  
#   # Record from SB3 model
#   python scripts/record_gif_after_training.py --model models/ppo/ppo_default.zip --output demo.gif
  
#   # Record with specific start position
#   python scripts/record_gif_after_training.py --model models/ppo/ppo_improved.pt --x 8.0 --y 8.0 --angle 0 --output corner.gif
  
#   # Record from multiple start positions
#   python scripts/record_gif_after_training.py --model models/ppo/ppo_improved.pt --multi --output-dir gifs/
  
#   # List available models
#   python scripts/record_gif_after_training.py --list-models
#         """
#     )
    
#     parser.add_argument("--model", "-m", type=str, help="Path to model file (.pt or .zip)")
#     parser.add_argument("--output", "-o", type=str, default="recording.gif", help="Output GIF path")
#     parser.add_argument("--output-dir", type=str, default="gifs", help="Output directory for multiple GIFs")
#     parser.add_argument("--fps", type=int, default=30, help="Frames per second")
#     parser.add_argument("--max-steps", type=int, default=600, help="Maximum steps per episode")
#     parser.add_argument("--x", type=float, help="Start X position")
#     parser.add_argument("--y", type=float, help="Start Y position")
#     parser.add_argument("--angle", type=float, help="Start angle (degrees)")
#     parser.add_argument("--multi", action="store_true", help="Record from multiple start positions")
#     parser.add_argument("--list-models", "-l", action="store_true", help="List available models")
#     parser.add_argument("--no-deterministic", action="store_true", help="Use stochastic policy")
    
#     args = parser.parse_args()
    
#     # List models
#     if args.list_models:
#         list_available_models()
#         return
    
#     # Check if model path is provided
#     if not args.model:
#         parser.print_help()
#         return
    
#     # Load model
#     model, model_type = load_model(args.model)
#     if model is None:
#         return
    
#     deterministic = not args.no_deterministic
    
#     # Record multiple GIFs
#     if args.multi:
#         record_multiple_gifs(model, model_type, args.output_dir, args.fps, args.max_steps)
#         return
    
#     # Set start position
#     start_position = None
#     if args.x is not None and args.y is not None and args.angle is not None:
#         start_position = (args.x, args.y, np.radians(args.angle))
#     elif args.x is not None or args.y is not None or args.angle is not None:
#         print("Error: To set start position, provide --x, --y, and --angle all together")
#         return
    
#     # Record single GIF
#     record_gif(model, model_type, args.output, args.fps, args.max_steps, start_position, deterministic)


# if __name__ == "__main__":
#     main()

# scripts/record_gif_after_training.py
"""
Record GIF from trained model without modifying training code.

Supports:
  - Custom PPO models (.pt)
  - Stable-Baselines3 models (.zip)
  - Custom SAC/TD3 models (.pt)

Usage:
  # Record from custom PPO model
  python scripts/record_gif_after_training.py --model models/ppo/ppo_improved.pt --output demo.gif
  
  # Record from SB3 model
  python scripts/record_gif_after_training.py --model models/ppo/ppo_improved.zip --output demo.gif
  
  # Record with specific start position
  python scripts/record_gif_after_training.py --model models/ppo/ppo_improved.pt --x 8.0 --y 8.0 --angle 0 --output corner.gif
  
  # Record from multiple start positions
  python scripts/record_gif_after_training.py --model models/ppo/ppo_improved.pt --multi --output-dir gifs/
  
  # List available models
  python scripts/record_gif_after_training.py --list-models
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import DiffDriveEnv


def find_models(model_dir="models"):
    """Find all available models in the directory."""
    models = []
    model_path = Path(model_dir)
    
    if model_path.exists():
        # Find .pt files (custom models)
        for file in model_path.glob("**/*.pt"):
            models.append({
                'path': str(file),
                'name': file.stem,
                'type': 'Custom PPO/SAC/TD3',
                'algo': file.parent.name if file.parent.name != 'models' else 'unknown'
            })
        
        # Find .zip files (Stable-Baselines3 models)
        for file in model_path.glob("**/*.zip"):
            models.append({
                'path': str(file),
                'name': file.stem,
                'type': 'Stable-Baselines3',
                'algo': file.parent.name if file.parent.name != 'models' else 'unknown'
            })
    
    return models


def load_model(model_path):
    """Load model based on file extension."""
    # Remove any extra .zip if present
    if model_path.endswith('.zip.zip'):
        model_path = model_path.replace('.zip.zip', '.zip')
    
    if model_path.endswith('.zip'):
        return load_sb3_model(model_path)
    elif model_path.endswith('.pt'):
        return load_custom_model(model_path)
    else:
        print(f"Unsupported model format: {model_path}")
        return None, None


def load_sb3_model(model_path):
    """Load Stable-Baselines3 model."""
    try:
        from stable_baselines3 import PPO, SAC, TD3
        
        # Check if file exists
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return None, None
        
        # Try different algorithms
        try:
            model = PPO.load(model_path)
            print(f"Loaded SB3 PPO model: {model_path}")
            return model, 'sb3'
        except Exception as e1:
            try:
                model = SAC.load(model_path)
                print(f"Loaded SB3 SAC model: {model_path}")
                return model, 'sb3'
            except Exception as e2:
                try:
                    model = TD3.load(model_path)
                    print(f"Loaded SB3 TD3 model: {model_path}")
                    return model, 'sb3'
                except Exception as e3:
                    print(f"Failed to load SB3 model: {e3}")
                    return None, None
    except ImportError:
        print("Error: stable-baselines3 not installed")
        return None, None


def load_custom_model(model_path):
    """Load custom PyTorch model."""
    try:
        # Try to load as PPO model first
        from models.ppo_model import PPOModel
        try:
            model = PPOModel.load(model_path)
            print(f"Loaded custom PPO model: {model_path}")
            return model, 'custom'
        except Exception as e1:
            pass
        
        # Try other custom models
        try:
            from models.sac_model import SACModel
            model = SACModel.load(model_path)
            print(f"Loaded custom SAC model: {model_path}")
            return model, 'custom'
        except:
            pass
        
        try:
            from models.td3_model import TD3Model
            model = TD3Model.load(model_path)
            print(f"Loaded custom TD3 model: {model_path}")
            return model, 'custom'
        except:
            pass
        
        print(f"Could not load custom model: {model_path}")
        return None, None
        
    except ImportError as e:
        print(f"Error loading custom model: {e}")
        return None, None


def record_gif(model, model_type, output_path, fps=30, max_steps=600, 
               start_position=None, deterministic=True):
    """Record a GIF from a trained model."""
    try:
        import imageio
    except ImportError:
        print("Error: imageio not installed. Install with: pip install imageio imageio-ffmpeg")
        return False
    
    # Create environment
    env = DiffDriveEnv(render_mode='rgb_array')
    
    # Set start position
    if start_position is not None:
        env.x, env.y, env.theta = start_position
        env.trajectory = [(env.x, env.y, env.theta)]
        print(f"Recording from custom position: x={env.x:.2f}, y={env.y:.2f}, angle={np.degrees(env.theta):.1f}°")
    else:
        env.reset()
        print(f"Recording from random start: x={env.x:.2f}, y={env.y:.2f}, angle={np.degrees(env.theta):.1f}°")
    
    # Record frames
    frames = []
    
    # Get initial observation
    if hasattr(env, '_obs'):
        obs = env._obs()
    else:
        x, y, theta = env.get_state()
        dist = np.hypot(x, y)
        obs = np.array([
            x / env.FIELD,
            y / env.FIELD,
            np.cos(theta),
            np.sin(theta),
            dist / (env.FIELD * np.sqrt(2)),
        ], dtype=np.float32)
    
    done = False
    step = 0
    
    print("Recording...", end="", flush=True)
    
    while not done and step < max_steps:
        # Get action from model
        try:
            if model_type == 'sb3':
                action, _ = model.predict(obs, deterministic=deterministic)
            else:  # custom model
                action = model.predict(obs, deterministic=deterministic)
        except Exception as e:
            print(f"\nError getting action: {e}")
            break
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        
        # Capture frame
        try:
            frame = env.render(mode='rgb_array')
            if frame is not None:
                frames.append(frame)
        except Exception as e:
            print(f"\nWarning: Could not render frame: {e}")
        
        if step % 100 == 0:
            print(".", end="", flush=True)
    
    print(" done!")
    
    # Save GIF
    if frames:
        # Create output directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        try:
            with imageio.get_writer(output_path, mode='I', fps=fps, loop=0) as writer:
                for frame in frames:
                    writer.append_data(frame)
            print(f"GIF saved: {output_path}")
            print(f"  Frames: {len(frames)}")
            print(f"  Duration: {len(frames)/fps:.1f} seconds")
            print(f"  Steps: {step}")
            print(f"  Final distance: {info['dist']:.3f} m")
            print(f"  Success: {info.get('success', False)}")
            return True
        except Exception as e:
            print(f"Error saving GIF: {e}")
            return False
    else:
        print("Warning: No frames captured!")
        return False
    
    env.close()


def record_multiple_gifs(model, model_type, output_dir, fps=30, max_steps=600):
    """Record GIFs from multiple start positions."""
    os.makedirs(output_dir, exist_ok=True)
    
    start_positions = [
        (None, "random"),
        (8.0, 8.0, 0.0, "far_corner_forward"),
        (8.0, 8.0, 2.0, "far_corner_wrong"),
        (-8.0, -8.0, 0.0, "opposite_corner"),
        (5.0, -5.0, 1.57, "side_facing_away"),
        (0.0, 8.0, 3.14, "top_facing_down"),
        (-5.0, 5.0, -1.57, "bottom_facing_up"),
        (8.0, 0.0, 1.57, "right_facing_up"),
        (-8.0, 0.0, -1.57, "left_facing_down"),
    ]
    
    successful = 0
    for pos in start_positions:
        if len(pos) == 4:
            x, y, angle, name = pos
            start_pos = (x, y, angle)
            output_path = os.path.join(output_dir, f"{name}.gif")
        else:
            start_pos = None
            output_path = os.path.join(output_dir, f"random_start.gif")
        
        print(f"\n{'='*50}")
        if record_gif(model, model_type, output_path, fps, max_steps, start_pos):
            successful += 1
    
    print(f"\n{'='*50}")
    print(f"Recorded {successful}/{len(start_positions)} GIFs")
    print(f"Saved to: {output_dir}")


def list_available_models():
    """List all available models."""
    print("\nAvailable models:")
    models = find_models()
    
    if not models:
        print("  No models found in models/ directory")
        return
    
    # Group by algorithm
    models_by_algo = {}
    for model in models:
        algo = model['algo'].upper()
        if algo not in models_by_algo:
            models_by_algo[algo] = []
        models_by_algo[algo].append(model)
    
    for algo, models_list in models_by_algo.items():
        print(f"\n{algo}:")
        for model in models_list:
            print(f"  • {model['name']} ({model['type']})")
            print(f"    Path: {model['path']}")


def main():
    parser = argparse.ArgumentParser(
        description="Record GIF from trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record from custom PPO model
  python scripts/record_gif_after_training.py --model models/ppo/ppo_improved.pt --output demo.gif
  
  # Record from SB3 model
  python scripts/record_gif_after_training.py --model models/ppo/ppo_improved.zip --output demo.gif
  
  # Record with specific start position
  python scripts/record_gif_after_training.py --model models/ppo/ppo_improved.pt --x 8.0 --y 8.0 --angle 0 --output corner.gif
  
  # Record from multiple start positions
  python scripts/record_gif_after_training.py --model models/ppo/ppo_improved.pt --multi --output-dir gifs/
  
  # List available models
  python scripts/record_gif_after_training.py --list-models
        """
    )
    
    parser.add_argument("--model", "-m", type=str, help="Path to model file (.pt or .zip)")
    parser.add_argument("--output", "-o", type=str, default="recording.gif", help="Output GIF path")
    parser.add_argument("--output-dir", type=str, default="gifs", help="Output directory for multiple GIFs")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--max-steps", type=int, default=600, help="Maximum steps per episode")
    parser.add_argument("--x", type=float, help="Start X position")
    parser.add_argument("--y", type=float, help="Start Y position")
    parser.add_argument("--angle", type=float, help="Start angle (degrees)")
    parser.add_argument("--multi", action="store_true", help="Record from multiple start positions")
    parser.add_argument("--list-models", "-l", action="store_true", help="List available models")
    parser.add_argument("--no-deterministic", action="store_true", help="Use stochastic policy")
    
    args = parser.parse_args()
    
    # List models
    if args.list_models:
        list_available_models()
        return
    
    # Check if model path is provided
    if not args.model:
        parser.print_help()
        return
    
    # Clean model path
    model_path = args.model
    if model_path.endswith('.zip.zip'):
        model_path = model_path.replace('.zip.zip', '.zip')
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("\nAvailable models:")
        list_available_models()
        return
    
    # Load model
    model, model_type = load_model(model_path)
    if model is None:
        return
    
    deterministic = not args.no_deterministic
    
    # Record multiple GIFs
    if args.multi:
        record_multiple_gifs(model, model_type, args.output_dir, args.fps, args.max_steps)
        return
    
    # Set start position
    start_position = None
    if args.x is not None and args.y is not None and args.angle is not None:
        start_position = (args.x, args.y, np.radians(args.angle))
    elif args.x is not None or args.y is not None or args.angle is not None:
        print("Error: To set start position, provide --x, --y, and --angle all together")
        return
    
    # Record single GIF
    record_gif(model, model_type, args.output, args.fps, args.max_steps, start_position, deterministic)


if __name__ == "__main__":
    main()