from src import SimilarityBasedUndersampler
from src.utils import load_config, check_system_requirements

def main():
    # Check system requirements
    sys_info = check_system_requirements()
    print("System Information:")
    print(f"GPU Available: {sys_info['gpu']}")
    if sys_info['gpu']:
        print(f"GPU Device: {sys_info['gpu_name']}")
    print(f"PyTorch Version: {sys_info['torch_version']}")
    
    # Load configuration
    config = load_config('config/config.yaml')
    
    # Initialize undersampler
    undersampler = SimilarityBasedUndersampler(config)
    
    # Define directories
    input_dir = 'OUTPUT_DATASET'
    output_dir = 'UNDERSAMPLED_DATASET'
    
    # Perform undersampling
    try:
        stats = undersampler.undersample_dataset(input_dir, output_dir)
        
        print("\nUndersampling Complete!")
        print(f"Original images: {stats['basic']['n_original_samples']}")
        print(f"Similar groups found: {stats['basic']['n_similar_groups']}")
        print(f"Average group size: {stats['basic']['avg_group_size']:.2f}")
        print(f"Execution time: {stats['performance']['execution_time_formatted']}")
        print("\nCheck the output directory for:")
        print("- Undersampled images and labels")
        print("- Visualizations")
        print("- Detailed HTML report")
        
    except Exception as e:
        print(f"Error during undersampling: {str(e)}")
        print("Check the logs directory for details")

if __name__ == "__main__":
    main()