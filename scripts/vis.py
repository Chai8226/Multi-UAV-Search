import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import argparse

def create_probability_animation(file_path, save_animation=False, fps=10):
    """
    Loads a JSON file and creates a time-aware 2D heatmap animation
    by resampling the data to a constant frame rate.
    """
    print(f"Loading file: {file_path}")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found '{file_path}'")
        return
    except json.JSONDecodeError as e:
        print(f"Error: File '{file_path}' is not a valid JSON file. {e}")
        return

    metadata = data.get('metadata', {})
    history = data.get('history', [])

    if not metadata or not history:
        print("Error: JSON file is missing 'metadata' or 'history' section.")
        return

    # 1. Reconstruct grid structure from metadata (same as before)
    cell_size = metadata['cell_size']
    cell_centers = {int(k): v for k, v in metadata['cell_centers'].items()}
    if not cell_centers:
        print("Error: No grid cell centers found in metadata.")
        return

    all_x = [center[0] for center in cell_centers.values()]
    all_y = [center[1] for center in cell_centers.values()]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    grid_dim_x = int(round((max_x - min_x) / cell_size[0]) + 1)
    grid_dim_y = int(round((max_y - min_y) / cell_size[1]) + 1)
    print(f"Grid reconstructed: {grid_dim_x} x {grid_dim_y} cells")

    id_to_indices = {}
    for grid_id, center in cell_centers.items():
        ix = int(round((center[0] - min_x) / cell_size[0]))
        iy = int(round((center[1] - min_y) / cell_size[1]))
        id_to_indices[grid_id] = (ix, iy)

    # 2. Resample data to a constant FPS to avoid time jumps
    print(f"Resampling data to a constant frame rate of {fps} FPS...")
    
    resampled_frames = []
    resampled_timestamps = []
    
    start_time = history[0]['timestamp']
    end_time = history[-1]['timestamp']
    total_duration = end_time - start_time
    
    # Create a list of playback timestamps at the desired FPS
    playback_timestamps = np.arange(0, total_duration, 1.0 / fps)
    
    original_data_idx = 0
    
    # Pre-build all original frames to avoid redundant work in the loop
    original_frames = []
    for record in history:
        frame_grid = np.zeros((grid_dim_y, grid_dim_x))
        for grid_id_str, prob in record['probabilities'].items():
            grid_id = int(grid_id_str)
            if grid_id in id_to_indices:
                ix, iy = id_to_indices[grid_id]
                frame_grid[iy, ix] = prob
        original_frames.append(frame_grid)

    # Interpolate frames
    for t_playback in playback_timestamps:
        # Find the latest original data point that is not in the future
        while (original_data_idx + 1 < len(history) and
               history[original_data_idx + 1]['timestamp'] - start_time <= t_playback):
            original_data_idx += 1
            
        resampled_frames.append(original_frames[original_data_idx])
        resampled_timestamps.append(t_playback)

    print(f"Resampling complete. Total frames for animation: {len(resampled_frames)}")

    # 3. Create and play the animation
    fig, ax = plt.subplots(figsize=(10, 10 * (grid_dim_y / max(1, grid_dim_x))))
    
    norm = Normalize(vmin=0, vmax=1.0)
    im = ax.imshow(resampled_frames[0], cmap='viridis', origin='lower', norm=norm, 
                   extent=[min_x - cell_size[0]/2, max_x + cell_size[0]/2, 
                           min_y - cell_size[1]/2, max_y + cell_size[1]/2])
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Target Probability')
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white',
                          bbox=dict(facecolor='black', alpha=0.5))

    def update(frame):
        im.set_data(resampled_frames[frame])
        time_text.set_text(f'Time: {resampled_timestamps[frame]:.2f} s')
        return im, time_text

    # The interval is now determined by the desired FPS
    interval_ms = 1000 / fps
    ani = animation.FuncAnimation(fig, update, frames=len(resampled_frames), blit=True, interval=interval_ms)

    ax.set_title('Global Grid Target Probability Over Time (Time-Aware)')
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    
    if save_animation:
        output_filename = './probability_animation_time_aware.mp4'
        print(f"Saving animation to {output_filename}... (This may take a moment)")
        try:
            ani.save(output_filename, writer='ffmpeg', fps=fps)
            print("Animation saved successfully!")
        except Exception as e:
            print(f"\nError: Could not save animation. Please ensure ffmpeg is installed on your system.")
            print(f"Details: {e}")
            print("Showing animation instead.")
            plt.show()
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a time-aware 2D heatmap animation of target probabilities.")
    parser.add_argument(
        '--file', 
        type=str, 
        default='/home/chai/Code/SCAR/search/Multi-UAV-Search/src/falcon_planner/exploration_manager/json/target_probability_with_metadata_1.json',
        help='Path to the JSON file containing metadata and probability history.'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='If set, save the animation to an MP4 file instead of displaying it.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Playback frame rate (Frames Per Second). Higher value means faster playback.'
    )
    
    args = parser.parse_args()
    # For compatibility, we check if --interval was passed in a different context, but prefer --fps
    create_probability_animation(args.file, args.save, args.fps)