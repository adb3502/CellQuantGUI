# CellQuantGUI - Complete System Documentation

## 📋 Overview
This document provides a comprehensive hierarchical listing of all scripts, classes, functions, and methods in the CellQuantGUI system.

---

## 📁 File Structure & Components

### 1. **cellquant_main.py** - Main GUI Application
*Core GUI application with experiment management and user interface*

#### Classes:
- **`ChannelInfo`** *(dataclass)*
  - Represents information about a microscopy channel
  - Attributes: `name`, `type`, `wavelength`, `purpose`

- **`Condition`** *(dataclass)*
  - Represents an experimental condition
  - Attributes: `name`, `directory`, `description`, `replicate_count`, `channels`
  - Methods:
    - `__post_init__()` - Initialize channels list if None

- **`ExperimentConfig`** *(dataclass)*
  - Complete experiment configuration
  - Attributes: `name`, `output_directory`, `conditions`, `analysis_parameters`, `created_date`
  - Methods:
    - `save_config(filepath)` - Save configuration to JSON file
    - `load_config(filepath)` *(classmethod)* - Load configuration from JSON file

- **`ImageProcessor`**
  - Handles image loading and basic processing operations
  - Methods:
    - `__init__()` - Initialize with supported formats
    - `discover_images(directory)` - Discover all supported image files in directory
    - `group_multichannel_images(image_paths)` - Group single-channel images by base name
    - `_extract_base_name(filename)` *(private)* - Extract base image name, removing channel identifiers

- **`SegmentationEngine`**
  - Cellpose integration for cell segmentation
  - Methods:
    - `__init__()` - Initialize and check Cellpose installation
    - `_check_cellpose_installation()` *(private)* - Verify Cellpose is available
    - `segment_cells(image, model_type, diameter)` - Perform cell segmentation using Cellpose
    - `_masks_to_rois(masks)` *(private)* - Convert segmentation masks to ROI coordinates

- **`QuantificationEngine`**
  - Handles CTCF and other quantitative measurements
  - Methods:
    - `__init__()` - Initialize quantification engine
    - `calculate_ctcf(image, roi_coords, background_value)` - Calculate Corrected Total Cell Fluorescence
    - `estimate_background(image, masks, method)` - Estimate background fluorescence

- **`MainApplication`**
  - Main GUI application controller
  - Methods:
    - `__init__()` - Initialize GUI and core components
    - `setup_gui()` - Initialize the GUI components
    - `setup_experiment_tab()` - Setup experiment configuration tab
    - `setup_analysis_tab()` - Setup analysis configuration and execution tab
    - `setup_results_tab()` - Setup results viewing and export tab
    - `new_experiment()` - Create a new experiment
    - `start_analysis()` - Start the analysis pipeline
    - `run()` - Start the application

---

### 2. **analysis_pipeline.py** - Core Analysis Engine
*Handles image processing, segmentation, and quantification*

#### Classes:
- **`AnalysisPipeline`**
  - Core analysis pipeline for quantitative microscopy
  - Methods:
    - `__init__(experiment_config, progress_callback)` - Initialize pipeline with configuration
    - `_initialize_cellpose()` *(private)* - Initialize Cellpose model
    - `run_analysis()` - Run the complete analysis pipeline
    - `_validate_configuration()` *(private)* - Validate analysis configuration
    - `_validate_condition(condition)` *(private)* - Validate a single condition configuration
    - `_process_condition(condition)` *(private)* - Process a single experimental condition
    - `_discover_images(directory)` *(private)* - Discover and group images in directory
    - `_is_multichannel_tiff(filepath)` *(private)* - Check if TIFF file has multiple channels
    - `_group_single_channel_images(image_paths)` *(private)* - Group single-channel images by base name
    - `_get_image_group_name(image_group)` *(private)* - Get display name for image group
    - `_load_image_group(image_group, channels)` *(private)* - Load image data according to channel configuration
    - `_load_single_image(filepath)` *(private)* - Load a single image file
    - `_load_multichannel_tiff(filepath)` *(private)* - Load multichannel TIFF file
    - `_segment_cells(image_data, condition)` *(private)* - Perform cell segmentation using Cellpose
    - `_select_segmentation_channel(image_data, channels)` *(private)* - Select the best channel for segmentation
    - `_masks_to_rois(masks)` *(private)* - Convert segmentation masks to ROI dictionaries
    - `_quantify_fluorescence(image_data, masks, cell_rois, condition)` *(private)* - Quantify fluorescence for each cell
    - `_estimate_background(image, masks)` *(private)* - Estimate background fluorescence
    - `_calculate_ctcf(image, coords, background, nuclear_only, masks)` *(private)* - Calculate Corrected Total Cell Fluorescence
    - `_calculate_nuclear_measurements(image, coords, background)` *(private)* - Calculate nuclear-specific measurements
    - `_summarize_condition_data(cell_data)` *(private)* - Generate summary statistics for a condition
    - `_generate_summary_statistics()` *(private)* - Generate overall experiment summary statistics
    - `_compare_conditions()` *(private)* - Compare measurements between conditions
    - `_export_results()` *(private)* - Export analysis results
    - `_export_cell_data_csv(output_dir)` *(private)* - Export detailed cell data as CSV
    - `_export_condition_summaries(output_dir)` *(private)* - Export condition summaries as CSV
    - `_update_progress(message, percentage)` *(private)* - Update progress callback

- **`SegmentationEngine`**
  - Cellpose integration for cell segmentation - Updated for v4.0.4+
  - Methods:
    - `__init__()` - Initialize segmentation engine
    - `_check_cellpose_installation()` *(private)* - Verify Cellpose is available
    - `segment_cells(image, model_type, diameter)` - Perform cell segmentation using Cellpose v4.0.4+
    - `_fallback_segmentation(image)` *(private)* - Fallback segmentation when Cellpose fails
    - `_masks_to_rois(masks)` *(private)* - Convert segmentation masks to ROI coordinates

- **`AnalysisParameters`**
  - Default analysis parameters and validation
  - Class Attributes: `DEFAULT_PARAMS` *(dict)*
  - Methods:
    - `get_default_config()` *(classmethod)* - Get default analysis configuration
    - `validate_parameters(params)` *(classmethod)* - Validate and fill in missing parameters

---

### 3. **channel_configurator.py** - Channel Setup Dialogs
*Handles assignment of channel types and quantification parameters*

#### Classes:
- **`ChannelConfigurator`**
  - Dialog for configuring channel information and assignments
  - Methods:
    - `__init__(parent, condition_name, image_paths)` - Initialize channel configurator dialog
    - `setup_dialog()` - Setup the dialog interface
    - `setup_scrollable_frame(parent)` - Create scrollable frame for channel configuration
    - `auto_detect_channels()` - Automatically detect and configure channels based on filenames
    - `is_multichannel_tiff(filepath)` - Check if TIFF file is multichannel
    - `detect_multichannel_tiff(filepath)` - Detect channels in multichannel TIFF
    - `detect_single_channel_images()` - Detect channels from multiple single-channel images
    - `guess_channel_type(name)` - Guess channel type based on name
    - `guess_channel_purpose(name)` - Guess channel purpose based on name
    - `should_quantify_by_default(name)` - Determine if channel should be quantified by default
    - `is_likely_nuclear(name)` - Determine if signal is likely nuclear-only
    - `guess_wavelength(name)` - Guess wavelength based on common naming conventions
    - `refresh_channel_display()` - Refresh the channel configuration display
    - `create_channel_row(row_idx, channel)` - Create widget row for a single channel
    - `add_channel()` - Add a new channel configuration
    - `remove_channel()` - Remove selected channel
    - `preview_channel(channel_idx)` - Preview the selected channel
    - `load_template()` - Load channel configuration template
    - `save_template()` - Save current configuration as template
    - `collect_channel_data()` - Collect channel data from UI widgets
    - `validate_configuration()` - Validate the channel configuration
    - `accept()` - Accept the configuration and close dialog
    - `cancel()` - Cancel and close dialog
    - `show()` - Show the dialog and return the result

- **`ConditionManager`**
  - Manages experimental conditions and their configurations
  - Methods:
    - `__init__(parent_app)` - Initialize condition manager
    - `add_condition_dialog()` - Show dialog to add a new condition
    - `browse_directory(dir_var)` - Browse for directory
    - `scan_directory(directory_path)` - Scan directory for images and update list
    - `get_images_from_directory(directory)` - Get list of supported images from directory

---

### 4. **visualization_module.py** - Data Visualization and Plotting
*Generates publication-ready plots and visualizations*

#### Classes:
- **`VisualizationEngine`**
  - Handles all visualization and plotting functionality
  - Methods:
    - `__init__(output_dir)` - Initialize visualization engine with output directory
    - `create_condition_comparison_plots(results)` - Create comprehensive condition comparison plots
    - `_create_condition_bar_plot(channel_name, condition_data)` *(private)* - Create bar plot comparing conditions for a specific channel
    - `_create_condition_box_plot(channel_name, condition_data, results)` *(private)* - Create box plot comparing conditions using individual cell data
    - `_create_experiment_summary_plot(results)` *(private)* - Create overall experiment summary visualization
    - `_plot_cell_counts(ax, results)` *(private)* - Plot cell counts per condition
    - `_plot_image_counts(ax, results)` *(private)* - Plot image counts per condition
    - `_plot_cell_area_distribution(ax, results)` *(private)* - Plot cell area distribution across conditions
    - `_plot_processing_info(ax, results)` *(private)* - Plot processing information
    - `create_segmentation_overlay(image, masks, output_filename)` - Create segmentation overlay visualization
    - `create_correlation_matrix(results)` - Create correlation matrix of all quantified channels
    - `create_statistical_report(results)` - Create comprehensive statistical report
    - `_perform_statistical_tests(results)` *(private)* - Perform statistical tests between conditions
    - `_create_summary_table(ax, results)` *(private)* - Create summary statistics table
    - `_create_statistical_test_table(ax, statistical_tests)` *(private)* - Create statistical test results table
    - `_create_effect_size_plots(ax1, ax2, statistical_tests)` *(private)* - Create effect size visualization plots
    - `_create_distribution_comparison(ax, results)` *(private)* - Create distribution comparison plot
    - `_create_power_analysis_summary(ax, statistical_tests)` *(private)* - Create power analysis summary

- **`InteractiveVisualization`**
  - Interactive visualization components for GUI integration
  - Methods:
    - `__init__(master_widget)` - Initialize interactive visualization
    - `create_embedded_plot(figure, title)` - Create embedded matplotlib plot in tkinter
    - `create_live_progress_plot()` - Create live progress visualization during analysis

#### Functions:
- `integrate_visualization_with_gui(main_app)` - Integrate visualization capabilities with main GUI application
- `generate_all_plots(viz_engine, results)` - Generate all standard plots
- `create_custom_plot_dialog(viz_engine)` - Create dialog for custom plot creation

---

### 5. **installation_setup.py** - Installation and Setup Scripts
*Comprehensive setup for CellQuantGUI system*

#### Classes:
- **`CellQuantInstaller`**
  - Handles installation and setup of CellQuantGUI system
  - Attributes: `dependencies` *(dict)*, `installation_status` *(dict)*
  - Methods:
    - `__init__()` - Initialize installer with system information
    - `setup_logging()` - Setup installation logging
    - `check_python_version()` - Check if Python version is compatible
    - `check_system_requirements()` - Check system-specific requirements
    - `check_pip()` - Check if pip is available
    - `check_git()` - Check if git is available (optional)
    - `check_display()` - Check if display is available for GUI
    - `check_memory()` - Check if system has adequate memory
    - `check_disk_space()` - Check available disk space
    - `check_dependencies()` - Check which dependencies are already installed
    - `install_dependencies(categories, force_reinstall)` - Install specified dependency categories
    - `install_package(package, version_req, force_reinstall)` - Install a single package
    - `setup_gpu_support()` - Setup GPU support for Cellpose if available
    - `setup_cuda_pytorch()` - Install CUDA-enabled PyTorch
    - `create_project_structure(project_dir)` - Create recommended project directory structure
    - `create_readme_files(project_dir)` - Create README files for project structure
    - `create_example_config(project_dir)` - Create example configuration file
    - `run_installation_tests()` - Run tests to verify installation
    - `test_basic_imports()` - Test basic package imports
    - `test_cellpose_functionality()` - Test Cellpose installation and basic functionality
    - `test_gui_components()` - Test GUI components
    - `test_image_processing()` - Test image processing capabilities
    - `test_analysis_pipeline()` - Test analysis pipeline components
    - `generate_installation_report()` - Generate comprehensive installation report

#### Functions:
- `run_interactive_installer()` - Run interactive installation process
- `create_batch_installer()` - Create batch installation scripts for different platforms

---

### 6. **template_manager.py** - Template and Configuration Management
*Handles saving, loading, and managing analysis templates*

#### Classes:
- **`AnalysisTemplate`** *(dataclass)*
  - Represents a reusable analysis template
  - Attributes: `name`, `description`, `category`, `version`, `created_date`, `modified_date`, `author`, `channel_template`, `analysis_parameters`, `file_patterns`, `instructions`, `tags`, `citation`, `doi`
  - Methods:
    - `to_dict()` - Convert template to dictionary
    - `from_dict(data)` *(classmethod)* - Create template from dictionary
    - `get_hash()` - Get hash of template for versioning

- **`TemplateManager`**
  - Manages analysis templates and configurations
  - Methods:
    - `__init__(templates_dir)` - Initialize template manager
    - `_create_builtin_templates()` *(private)* - Create built-in analysis templates
    - `_create_dna_damage_template()` *(private)* - Create DNA damage analysis template
    - `_create_mitochondrial_template()` *(private)* - Create mitochondrial analysis template
    - `_create_protein_localization_template()` *(private)* - Create protein localization template
    - `_create_cell_cycle_template()` *(private)* - Create cell cycle analysis template
    - `_create_apoptosis_template()` *(private)* - Create apoptosis analysis template
    - `load_all_templates()` - Load all available templates
    - `get_template(template_id)` - Get template by ID
    - `list_templates(category)` - List available templates, optionally filtered by category
    - `get_categories()` - Get list of available template categories
    - `save_template(template, user_template)` - Save template to file
    - `_save_template(template, target_dir, overwrite)` *(private)* - Internal method to save template
    - `load_template(filepath)` - Load template from file
    - `delete_template(template_id)` - Delete a user template
    - `duplicate_template(template_id, new_name)` - Duplicate an existing template
    - `create_template_from_config(config, template_name, template_description, category)` - Create a new template from an existing configuration
    - `apply_template_to_condition(template_id, condition_config)` - Apply template configuration to a condition
    - `get_template_suggestions(image_filenames)` - Suggest templates based on image filenames
    - `export_template(template_id, export_path)` - Export template to external file
    - `import_template(import_path)` - Import template from external file
    - `validate_template(template)` - Validate template configuration and return list of issues

- **`TemplateManagerGUI`**
  - GUI components for template management
  - Methods:
    - `__init__(parent, template_manager)` - Initialize template manager GUI
    - `create_template_selection_dialog()` - Create dialog for selecting analysis template

#### Functions:
- `demonstrate_template_system()` - Demonstrate the template management system

---

### 7. **file_management.py** - File Management and Batch Processing
*Handles file organization, batch operations, and data management*

#### Classes:
- **`FileInfo`** *(dataclass)*
  - Information about a microscopy file
  - Attributes: `path`, `size`, `modified_time`, `checksum`, `metadata`
  - Properties: `size_mb` - File size in megabytes

- **`ProcessingJob`** *(dataclass)*
  - Represents a batch processing job
  - Attributes: `job_id`, `name`, `status`, `created_time`, `start_time`, `end_time`, `progress`, `total_files`, `processed_files`, `failed_files`, `config`, `results_dir`, `log_file`, `error_message`

- **`FileManager`**
  - Comprehensive file management for microscopy data
  - Methods:
    - `__init__(base_directory)` - Initialize file manager
    - `scan_directory(directory, recursive, update_index)` - Scan directory for microscopy files
    - `_create_file_info(file_path)` *(private)* - Create FileInfo object for a file
    - `_calculate_checksum(file_path, chunk_size)` *(private)* - Calculate MD5 checksum of file
    - `_extract_metadata(file_path)` *(private)* - Extract metadata from image file
    - `_extract_tiff_metadata(file_path)` *(private)* - Extract TIFF-specific metadata
    - `organize_files(files, organization_scheme)` - Organize files according to specified scheme
    - `find_duplicate_files(files)` - Find duplicate files based on checksum
    - `validate_file_integrity(files)` - Validate file integrity by checking if files can be opened
    - `create_file_manifest(files, output_file)` - Create a manifest file listing all files and their metadata

- **`BatchProcessor`**
  - Handles batch processing of multiple experiments
  - Methods:
    - `__init__(max_workers)` - Initialize batch processor
    - `create_batch_job(name, configs, results_base_dir)` - Create a new batch processing job
    - `start_batch_job(job_id, progress_callback)` - Start executing a batch job
    - `_execute_batch_job(job, progress_callback)` *(private)* - Execute batch job in background thread
    - `_generate_batch_summary(job)` *(private)* - Generate summary report for batch job
    - `get_job_status(job_id)` - Get current status of a job
    - `cancel_job(job_id)` - Cancel a running job
    - `list_jobs()` - List all jobs
    - `cleanup_completed_jobs(max_age_days)` - Clean up old completed jobs

- **`DataArchiver`**
  - Handles data archiving and backup operations
  - Methods:
    - `__init__(archive_base_dir)` - Initialize data archiver
    - `create_archive(source_dir, archive_name, include_raw_images, include_results, compression)` - Create archive of experimental data
    - `restore_archive(archive_path, restore_dir)` - Restore archive to specified directory

- **`WatchdogMonitor`**
  - Monitors directories for new files and automatically processes them
  - Methods:
    - `__init__(watch_dir, config_template)` - Initialize watchdog monitor
    - `start_monitoring()` - Start monitoring directory for new files
    - `stop_monitoring()` - Stop monitoring directory
    - `_monitor_loop()` *(private)* - Main monitoring loop
    - `_scan_for_new_files()` *(private)* - Scan for new files in watch directory
    - `_group_files_by_experiment(files)` *(private)* - Group files by experiment based on directory structure
    - `_process_file_group(group_name, files)` *(private)* - Process a group of files

- **`FileManagerGUI`**
  - GUI components for file management
  - Methods:
    - `__init__(parent, file_manager)` - Initialize file manager GUI
    - `create_file_browser_dialog()` - Create file browser dialog for selecting files
    - `create_batch_processing_dialog(batch_processor)` - Create dialog for setting up batch processing

#### Functions:
- `demonstrate_file_management()` - Demonstrate file management capabilities

---

### 8. **quality_control.py** - Quality Control and Validation
*Ensures data quality and analysis reliability through comprehensive validation*

#### Classes:
- **`QualityMetric`** *(dataclass)*
  - Represents a quality control metric
  - Attributes: `name`, `value`, `threshold`, `status`, `description`, `recommendation`

- **`ValidationResult`** *(dataclass)*
  - Result of a validation check
  - Attributes: `check_name`, `status`, `message`, `details`, `timestamp`

- **`QualityReport`** *(dataclass)*
  - Comprehensive quality control report
  - Attributes: `experiment_name`, `timestamp`, `overall_status`, `metrics`, `validation_results`, `recommendations`, `data_summary`

- **`ImageQualityAnalyzer`**
  - Analyzes image quality metrics
  - Methods:
    - `__init__()` - Initialize image quality analyzer
    - `analyze_image_quality(image, channel_name)` - Analyze various image quality metrics
    - `_estimate_snr(image)` *(private)* - Estimate signal-to-noise ratio
    - `_calculate_rms_contrast(image)` *(private)* - Calculate RMS contrast
    - `_calculate_michelson_contrast(image)` *(private)* - Calculate Michelson contrast
    - `_calculate_gradient_magnitude(image)` *(private)* - Calculate average gradient magnitude (sharpness indicator)
    - `_calculate_laplacian_variance(image)` *(private)* - Calculate Laplacian variance (focus measure)
    - `_estimate_noise_level(image)` *(private)* - Estimate noise level using local standard deviation
    - `_calculate_saturation_percentage(image)` *(private)* - Calculate percentage of saturated pixels

- **`SegmentationQualityChecker`**
  - Validates cell segmentation quality
  - Methods:
    - `__init__()` - Initialize segmentation quality checker
    - `validate_segmentation(masks, original_image)` - Comprehensive segmentation validation
    - `_calculate_coverage_percentage(masks)` *(private)* - Calculate percentage of image covered by cells
    - `_calculate_cell_areas(masks)` *(private)* - Calculate area of each segmented cell
    - `_analyze_cell_morphology(masks)` *(private)* - Analyze morphological properties of segmented cells
    - `_assess_boundary_quality(masks, original_image)` *(private)* - Assess quality of segmentation boundaries
    - `_detect_segmentation_issues(masks, cell_areas)` *(private)* - Detect common segmentation issues

- **`DataQualityValidator`**
  - Validates quantitative measurement data quality
  - Methods:
    - `__init__()` - Initialize data quality validator
    - `validate_measurement_data(data, experiment_config)` - Comprehensive validation of measurement data
    - `_check_data_completeness(data)` *(private)* - Check for missing data and completeness
    - `_detect_outliers(data)` *(private)* - Detect outliers in measurement data
    - `_validate_distributions(data)` *(private)* - Validate measurement distributions
    - `_check_systematic_biases(data)` *(private)* - Check for systematic biases in the data
    - `_validate_condition_comparisons(data)` *(private)* - Validate comparisons between experimental conditions
    - `_check_measurement_consistency(data)` *(private)* - Check consistency of measurements

- **`QualityControlManager`**
  - Main quality control manager that coordinates all validation checks
  - Methods:
    - `__init__()` - Initialize quality control manager
    - `generate_quality_report(experiment_results, experiment_config)` - Generate comprehensive quality control report
    - `_generate_overall_metrics(experiment_results)` *(private)* - Generate overall quality metrics
    - `_generate_recommendations(validation_results, metrics)` *(private)* - Generate recommendations based on validation results
    - `_determine_overall_status(validation_results, metrics)` *(private)* - Determine overall quality status
    - `_create_data_summary(experiment_results)` *(private)* - Create summary of analyzed data
    - `save_quality_report(report, output_file)` - Save quality report to file
    - `create_quality_visualization(report, output_dir)` - Create visualizations for quality report

#### Functions:
- `demonstrate_quality_control()` - Demonstrate quality control capabilities

---

### 9. **example_usage.py** - Usage Examples and Integration
*Demonstrates how to use the CellQuantGUI system*

#### Functions:
- `setup_logging()` - Setup logging configuration
- `create_example_experiment_config()` - Create an example experiment configuration for testing
- `create_batch_analysis_config()` - Create configuration for batch analysis without GUI
- `run_gui_application()` - Run the main GUI application
- `run_batch_analysis()` - Run analysis in batch mode without GUI
- `run_configuration_example()` - Demonstrate experiment configuration creation and saving
- `demonstrate_channel_types()` - Demonstrate different channel configuration scenarios
- `check_dependencies()` - Check if required dependencies are installed
- `main()` - Main function to demonstrate different usage modes

---

## 📊 Summary Statistics

### **Total Components:**
- **9 Python Scripts**
- **35+ Classes**
- **200+ Methods/Functions**
- **15+ Data Classes**

### **Key Functional Areas:**
1. **GUI Framework** - Complete tkinter-based interface
2. **Image Processing** - Multi-format support with fallback mechanisms
3. **Cell Segmentation** - Cellpose integration with scikit-image fallback
4. **Quantitative Analysis** - CTCF calculation with statistical analysis
5. **Template System** - Pre-built assay configurations
6. **Batch Processing** - High-throughput analysis capabilities
7. **Quality Control** - Comprehensive validation and reporting
8. **Data Management** - File organization and archiving
9. **Visualization** - Publication-ready plots and reports

### **Extension Points:**
- **Foci Detection Module** - Ready for AutoFoci/FocAn integration
- **3D Analysis Module** - Z-stack processing framework
- **4D Analysis Module** - Time-lapse analysis capabilities
- **Custom Templates** - User-defined analysis workflows
- **Plugin Architecture** - Extensible analysis modules

---

## 🔧 Integration Notes

### **Dependencies:**
- **Core:** numpy, pandas, matplotlib, pillow, scikit-image
- **ML:** cellpose, torch, torchvision
- **GUI:** tkinter (built-in Python)
- **Optional:** opencv-python, napari, plotly, dash

### **Data Flow:**
```
Raw Images → Channel Configuration → Image Processing → 
Cell Segmentation → Fluorescence Quantification → 
Statistical Analysis → Results Export → Quality Control
```

### **File Relationships:**
- `cellquant_main.py` imports and orchestrates all other modules
- `analysis_pipeline.py` is the core engine used by GUI and batch modes
- `template_manager.py` provides pre-configured analysis workflows
- `quality_control.py` validates all analysis outputs
- `visualization_module.py` generates publication-ready figures

This comprehensive system provides a complete solution for quantitative microscopy analysis with professional-grade features, extensive validation, and user-friendly interfaces.