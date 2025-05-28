# File Management and Batch Processing Module
# Handles file organization, batch operations, and data management

import os
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterator
from datetime import datetime, timedelta
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass
from collections import defaultdict
import re

@dataclass
class FileInfo:
    """Information about a microscopy file"""
    path: Path
    size: int
    modified_time: datetime
    checksum: str
    metadata: Dict[str, Any]
    
    @property
    def size_mb(self) -> float:
        return self.size / (1024 * 1024)

@dataclass
class ProcessingJob:
    """Represents a batch processing job"""
    job_id: str
    name: str
    status: str  # 'pending', 'running', 'completed', 'failed', 'cancelled'
    created_time: datetime
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    progress: float
    total_files: int
    processed_files: int
    failed_files: int
    config: Dict[str, Any]
    results_dir: Path
    log_file: Path
    error_message: Optional[str] = None

class FileManager:
    """Comprehensive file management for microscopy data"""
    
    def __init__(self, base_directory: Path = None):
        self.base_directory = base_directory or Path.cwd()
        self.logger = logging.getLogger(__name__)
        
        # Supported file formats
        self.supported_extensions = {
            '.tif', '.tiff', '.png', '.jpg', '.jpeg', 
            '.lsm', '.czi', '.nd2', '.oib', '.vsi'
        }
        
        # File index for fast searching
        self.file_index = {}
        self.last_index_update = None
        
    def scan_directory(self, directory: Path, recursive: bool = True, 
                      update_index: bool = True) -> List[FileInfo]:
        """Scan directory for microscopy files"""
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        files = []
        scan_pattern = "**/*" if recursive else "*"
        
        self.logger.info(f"Scanning directory: {directory}")
        
        for file_path in directory.glob(scan_pattern):
            if not file_path.is_file():
                continue
                
            if file_path.suffix.lower() not in self.supported_extensions:
                continue
            
            try:
                file_info = self._create_file_info(file_path)
                files.append(file_info)
                
                if update_index:
                    self.file_index[str(file_path)] = file_info
                    
            except Exception as e:
                self.logger.warning(f"Error processing file {file_path}: {e}")
        
        self.logger.info(f"Found {len(files)} microscopy files")
        self.last_index_update = datetime.now()
        
        return files
    
    def _create_file_info(self, file_path: Path) -> FileInfo:
        """Create FileInfo object for a file"""
        
        stat = file_path.stat()
        
        # Calculate checksum
        checksum = self._calculate_checksum(file_path)
        
        # Extract metadata
        metadata = self._extract_metadata(file_path)
        
        return FileInfo(
            path=file_path,
            size=stat.st_size,
            modified_time=datetime.fromtimestamp(stat.st_mtime),
            checksum=checksum,
            metadata=metadata
        )
    
    def _calculate_checksum(self, file_path: Path, chunk_size: int = 8192) -> str:
        """Calculate MD5 checksum of file"""
        
        hash_md5 = hashlib.md5()
        
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.warning(f"Could not calculate checksum for {file_path}: {e}")
            return ""
    
    def _extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from image file"""
        
        metadata = {
            'filename': file_path.name,
            'extension': file_path.suffix.lower(),
            'directory': str(file_path.parent)
        }
        
        # Try to extract image-specific metadata
        try:
            if file_path.suffix.lower() in ['.tif', '.tiff']:
                metadata.update(self._extract_tiff_metadata(file_path))
            # Could add support for other formats here
        except Exception as e:
            self.logger.debug(f"Could not extract detailed metadata from {file_path}: {e}")
        
        return metadata
    
    def _extract_tiff_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract TIFF-specific metadata"""
        
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
            
            with Image.open(file_path) as img:
                metadata = {
                    'width': img.width,
                    'height': img.height,
                    'mode': img.mode,
                    'format': img.format
                }
                
                # Get number of frames for multichannel images
                if hasattr(img, 'n_frames'):
                    metadata['n_frames'] = img.n_frames
                
                # Extract EXIF data
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    if exif:
                        for tag_id, value in exif.items():
                            tag = TAGS.get(tag_id, tag_id)
                            metadata[f'exif_{tag}'] = value
                
                return metadata
                
        except Exception as e:
            self.logger.debug(f"Error extracting TIFF metadata: {e}")
            return {}
    
    def organize_files(self, files: List[FileInfo], organization_scheme: str = "date") -> Dict[str, List[FileInfo]]:
        """Organize files according to specified scheme"""
        
        organized = defaultdict(list)
        
        for file_info in files:
            if organization_scheme == "date":
                key = file_info.modified_time.strftime("%Y-%m-%d")
            elif organization_scheme == "size":
                if file_info.size_mb < 1:
                    key = "small (<1MB)"
                elif file_info.size_mb < 10:
                    key = "medium (1-10MB)"
                else:
                    key = "large (>10MB)"
            elif organization_scheme == "extension":
                key = file_info.path.suffix.lower()
            elif organization_scheme == "directory":
                key = str(file_info.path.parent)
            else:
                key = "all"
            
            organized[key].append(file_info)
        
        return dict(organized)
    
    def find_duplicate_files(self, files: List[FileInfo]) -> List[List[FileInfo]]:
        """Find duplicate files based on checksum"""
        
        checksum_groups = defaultdict(list)
        
        for file_info in files:
            if file_info.checksum:
                checksum_groups[file_info.checksum].append(file_info)
        
        # Return only groups with duplicates
        duplicates = [group for group in checksum_groups.values() if len(group) > 1]
        
        self.logger.info(f"Found {len(duplicates)} groups of duplicate files")
        return duplicates
    
    def validate_file_integrity(self, files: List[FileInfo]) -> List[FileInfo]:
        """Validate file integrity by checking if files can be opened"""
        
        corrupted_files = []
        
        for file_info in files:
            try:
                # Try to open the file
                if file_info.path.suffix.lower() in ['.tif', '.tiff']:
                    from PIL import Image
                    with Image.open(file_info.path) as img:
                        # Try to load the image data
                        img.load()
                elif file_info.path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    from PIL import Image
                    with Image.open(file_info.path) as img:
                        img.verify()
                
            except Exception as e:
                self.logger.warning(f"File integrity check failed for {file_info.path}: {e}")
                corrupted_files.append(file_info)
        
        return corrupted_files
    
    def create_file_manifest(self, files: List[FileInfo], output_file: Path):
        """Create a manifest file listing all files and their metadata"""
        
        manifest_data = {
            'created_time': datetime.now().isoformat(),
            'total_files': len(files),
            'total_size_mb': sum(f.size_mb for f in files),
            'files': []
        }
        
        for file_info in files:
            file_data = {
                'path': str(file_info.path),
                'size_mb': file_info.size_mb,
                'modified_time': file_info.modified_time.isoformat(),
                'checksum': file_info.checksum,
                'metadata': file_info.metadata
            }
            manifest_data['files'].append(file_data)
        
        with open(output_file, 'w') as f:
            json.dump(manifest_data, f, indent=2)
        
        self.logger.info(f"File manifest created: {output_file}")

class BatchProcessor:
    """Handles batch processing of multiple experiments"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        
        # Job management
        self.jobs = {}
        self.job_counter = 0
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Lock for thread-safe operations
        self.lock = threading.Lock()
    
    def create_batch_job(self, name: str, configs: List[Dict[str, Any]], 
                        results_base_dir: Path) -> str:
        """Create a new batch processing job"""
        
        with self.lock:
            self.job_counter += 1
            job_id = f"batch_{self.job_counter:04d}"
        
        # Create job directory
        job_dir = results_base_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = job_dir / "batch_processing.log"
        
        job = ProcessingJob(
            job_id=job_id,
            name=name,
            status='pending',
            created_time=datetime.now(),
            start_time=None,
            end_time=None,
            progress=0.0,
            total_files=len(configs),
            processed_files=0,
            failed_files=0,
            config={'configs': configs},
            results_dir=job_dir,
            log_file=log_file
        )
        
        self.jobs[job_id] = job
        self.logger.info(f"Created batch job: {job_id} with {len(configs)} configurations")
        
        return job_id
    
    def start_batch_job(self, job_id: str, progress_callback=None) -> bool:
        """Start executing a batch job"""
        
        job = self.jobs.get(job_id)
        if not job:
            self.logger.error(f"Job not found: {job_id}")
            return False
        
        if job.status != 'pending':
            self.logger.error(f"Job {job_id} is not in pending state")
            return False
        
        # Update job status
        job.status = 'running'
        job.start_time = datetime.now()
        
        # Submit job to thread pool
        future = self.executor.submit(self._execute_batch_job, job, progress_callback)
        
        self.logger.info(f"Started batch job: {job_id}")
        return True
    
    def _execute_batch_job(self, job: ProcessingJob, progress_callback=None):
        """Execute batch job in background thread"""
        
        try:
            from analysis_pipeline import AnalysisPipeline
            
            configs = job.config['configs']
            
            for i, config in enumerate(configs):
                if job.status == 'cancelled':
                    break
                
                try:
                    # Create individual output directory
                    config_name = config.get('experiment_name', f'config_{i+1}')
                    config_output_dir = job.results_dir / config_name
                    config_output_dir.mkdir(exist_ok=True)
                    
                    # Update config with job-specific output directory
                    config['output_directory'] = str(config_output_dir)
                    
                    # Create progress callback for this configuration
                    def config_progress_callback(message, percentage):
                        if progress_callback:
                            overall_progress = ((i + percentage/100) / len(configs)) * 100
                            progress_callback(f"Config {i+1}/{len(configs)}: {message}", overall_progress)
                    
                    # Run analysis
                    pipeline = AnalysisPipeline(config, config_progress_callback)
                    results = pipeline.run_analysis()
                    
                    job.processed_files += 1
                    
                except Exception as e:
                    self.logger.error(f"Error processing config {i+1} in job {job.job_id}: {e}")
                    job.failed_files += 1
                
                # Update progress
                job.progress = (i + 1) / len(configs) * 100
                
                if progress_callback:
                    progress_callback(f"Completed {i+1}/{len(configs)} configurations", job.progress)
            
            # Job completed
            job.status = 'completed' if job.failed_files == 0 else 'completed_with_errors'
            job.end_time = datetime.now()
            
            # Generate batch summary report
            self._generate_batch_summary(job)
            
            self.logger.info(f"Batch job {job.job_id} completed: {job.processed_files} successful, {job.failed_files} failed")
            
        except Exception as e:
            job.status = 'failed'
            job.error_message = str(e)
            job.end_time = datetime.now()
            self.logger.error(f"Batch job {job.job_id} failed: {e}")
    
    def _generate_batch_summary(self, job: ProcessingJob):
        """Generate summary report for batch job"""
        
        summary = {
            'job_id': job.job_id,
            'name': job.name,
            'status': job.status,
            'created_time': job.created_time.isoformat(),
            'start_time': job.start_time.isoformat() if job.start_time else None,
            'end_time': job.end_time.isoformat() if job.end_time else None,
            'duration_minutes': (job.end_time - job.start_time).total_seconds() / 60 if job.end_time and job.start_time else None,
            'total_configurations': job.total_files,
            'successful_configurations': job.processed_files,
            'failed_configurations': job.failed_files,
            'success_rate': job.processed_files / job.total_files * 100 if job.total_files > 0 else 0
        }
        
        summary_file = job.results_dir / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Batch summary saved: {summary_file}")
    
    def get_job_status(self, job_id: str) -> Optional[ProcessingJob]:
        """Get current status of a job"""
        return self.jobs.get(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status == 'running':
            job.status = 'cancelled'
            self.logger.info(f"Job {job_id} cancelled")
            return True
        
        return False
    
    def list_jobs(self) -> List[ProcessingJob]:
        """List all jobs"""
        return list(self.jobs.values())
    
    def cleanup_completed_jobs(self, max_age_days: int = 30):
        """Clean up old completed jobs"""
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        jobs_to_remove = []
        for job_id, job in self.jobs.items():
            if (job.status in ['completed', 'failed', 'cancelled'] and
                job.end_time and job.end_time < cutoff_date):
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.jobs[job_id]
            self.logger.info(f"Cleaned up old job: {job_id}")

class DataArchiver:
    """Handles data archiving and backup operations"""
    
    def __init__(self, archive_base_dir: Path = None):
        self.archive_base_dir = archive_base_dir or Path("archives")
        self.archive_base_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def create_archive(self, source_dir: Path, archive_name: str, 
                      include_raw_images: bool = True, 
                      include_results: bool = True,
                      compression: str = 'zip') -> Path:
        """Create archive of experimental data"""
        
        archive_dir = self.archive_base_dir / archive_name
        archive_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Creating archive: {archive_name}")
        
        # Copy directory structure
        if include_raw_images:
            raw_data_dir = archive_dir / "raw_data"
            if (source_dir / "data").exists():
                shutil.copytree(source_dir / "data", raw_data_dir)
        
        if include_results:
            results_dir = archive_dir / "results"
            if (source_dir / "results").exists():
                shutil.copytree(source_dir / "results", results_dir)
        
        # Copy configuration files
        configs_dir = archive_dir / "configs"
        configs_dir.mkdir(exist_ok=True)
        if (source_dir / "configs").exists():
            for config_file in (source_dir / "configs").glob("*.json"):
                shutil.copy2(config_file, configs_dir)
        
        # Create archive metadata
        metadata = {
            'archive_name': archive_name,
            'created_time': datetime.now().isoformat(),
            'source_directory': str(source_dir),
            'include_raw_images': include_raw_images,
            'include_results': include_results,
            'compression': compression
        }
        
        metadata_file = archive_dir / "archive_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Compress archive if requested
        if compression == 'zip':
            archive_file = self.archive_base_dir / f"{archive_name}.zip"
            shutil.make_archive(str(archive_file.with_suffix('')), 'zip', archive_dir)
            
            # Remove uncompressed directory
            shutil.rmtree(archive_dir)
            
            self.logger.info(f"Archive created: {archive_file}")
            return archive_file
        else:
            self.logger.info(f"Archive created: {archive_dir}")
            return archive_dir
    
    def restore_archive(self, archive_path: Path, restore_dir: Path):
        """Restore archive to specified directory"""
        
        if archive_path.suffix == '.zip':
            shutil.unpack_archive(archive_path, restore_dir)
        else:
            shutil.copytree(archive_path, restore_dir)
        
        self.logger.info(f"Archive restored to: {restore_dir}")

class WatchdogMonitor:
    """Monitors directories for new files and automatically processes them"""
    
    def __init__(self, watch_dir: Path, config_template: Dict[str, Any]):
        self.watch_dir = watch_dir
        self.config_template = config_template
        self.logger = logging.getLogger(__name__)
        
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Processed files tracking
        self.processed_files = set()
    
    def start_monitoring(self):
        """Start monitoring directory for new files"""
        
        if self.is_monitoring:
            self.logger.warning("Already monitoring directory")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info(f"Started monitoring directory: {self.watch_dir}")
    
    def stop_monitoring(self):
        """Stop monitoring directory"""
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("Stopped directory monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        
        while self.is_monitoring:
            try:
                # Scan for new files
                new_files = self._scan_for_new_files()
                
                if new_files:
                    self.logger.info(f"Found {len(new_files)} new files")
                    
                    # Group files by experiment/condition
                    file_groups = self._group_files_by_experiment(new_files)
                    
                    # Process each group
                    for group_name, files in file_groups.items():
                        self._process_file_group(group_name, files)
                
                # Wait before next scan
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _scan_for_new_files(self) -> List[Path]:
        """Scan for new files in watch directory"""
        
        new_files = []
        
        for file_path in self.watch_dir.rglob("*"):
            if not file_path.is_file():
                continue
            
            if file_path.suffix.lower() not in {'.tif', '.tiff', '.png', '.jpg'}:
                continue
            
            if str(file_path) not in self.processed_files:
                new_files.append(file_path)
        
        return new_files
    
    def _group_files_by_experiment(self, files: List[Path]) -> Dict[str, List[Path]]:
        """Group files by experiment based on directory structure"""
        
        groups = defaultdict(list)
        
        for file_path in files:
            # Use parent directory name as group identifier
            group_name = file_path.parent.name
            groups[group_name].append(file_path)
        
        return dict(groups)
    
    def _process_file_group(self, group_name: str, files: List[Path]):
        """Process a group of files"""
        
        try:
            # Create configuration for this group
            config = self.config_template.copy()
            config['experiment_name'] = f"auto_{group_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Update condition directory
            if 'conditions' in config and config['conditions']:
                config['conditions'][0]['directory'] = str(files[0].parent)
            
            # Run analysis
            from analysis_pipeline import AnalysisPipeline
            
            pipeline = AnalysisPipeline(config)
            results = pipeline.run_analysis()
            
            # Mark files as processed
            for file_path in files:
                self.processed_files.add(str(file_path))
            
            self.logger.info(f"Successfully processed file group: {group_name}")
            
        except Exception as e:
            self.logger.error(f"Error processing file group {group_name}: {e}")

# GUI Integration Components
class FileManagerGUI:
    """GUI components for file management"""
    
    def __init__(self, parent, file_manager: FileManager):
        self.parent = parent
        self.file_manager = file_manager
        self.logger = logging.getLogger(__name__)
    
    def create_file_browser_dialog(self) -> Optional[List[Path]]:
        """Create file browser dialog for selecting files"""
        
        import tkinter as tk
        from tkinter import ttk, filedialog
        
        # Use standard file dialog for now
        files = filedialog.askopenfilenames(
            title="Select Microscopy Images",
            filetypes=[
                ("TIFF files", "*.tif *.tiff"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All supported", "*.tif *.tiff *.png *.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )
        
        return [Path(f) for f in files] if files else None
    
    def create_batch_processing_dialog(self, batch_processor: BatchProcessor):
        """Create dialog for setting up batch processing"""
        
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        
        dialog = tk.Toplevel(self.parent)
        dialog.title("Batch Processing Setup")
        dialog.geometry("800x600")
        dialog.grab_set()
        
        # Main frame
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Job information
        info_frame = ttk.LabelFrame(main_frame, text="Job Information")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(info_frame, text="Job Name:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        job_name_var = tk.StringVar(value=f"Batch_Job_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        ttk.Entry(info_frame, textvariable=job_name_var, width=40).grid(row=0, column=1, padx=5, pady=5)
        
        # Configuration list
        config_frame = ttk.LabelFrame(main_frame, text="Configurations")
        config_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        config_listbox = tk.Listbox(config_frame, height=10)
        config_scrollbar = ttk.Scrollbar(config_frame, orient=tk.VERTICAL, command=config_listbox.yview)
        config_listbox.configure(yscrollcommand=config_scrollbar.set)
        
        config_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        config_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Configuration management buttons
        config_btn_frame = ttk.Frame(config_frame)
        config_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        configurations = []
        
        def add_config():
            config_files = filedialog.askopenfilenames(
                title="Select Configuration Files",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            for config_file in config_files:
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    configurations.append(config)
                    config_listbox.insert(tk.END, f"{len(configurations)}. {Path(config_file).name}")
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load configuration: {e}")
        
        def remove_config():
            selection = config_listbox.curselection()
            if selection:
                index = selection[0]
                configurations.pop(index)
                config_listbox.delete(index)
                
                # Update list display
                config_listbox.delete(0, tk.END)
                for i, config in enumerate(configurations):
                    name = config.get('experiment_name', f'Config {i+1}')
                    config_listbox.insert(tk.END, f"{i+1}. {name}")
        
        ttk.Button(config_btn_frame, text="Add Config Files", command=add_config).pack(side=tk.LEFT, padx=2)
        ttk.Button(config_btn_frame, text="Remove Selected", command=remove_config).pack(side=tk.LEFT, padx=2)
        
        # Results directory
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(results_frame, text="Results Directory:").pack(side=tk.LEFT)
        results_dir_var = tk.StringVar(value=str(Path.cwd() / "batch_results"))
        ttk.Entry(results_frame, textvariable=results_dir_var, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(results_frame, text="Browse", 
                  command=lambda: results_dir_var.set(filedialog.askdirectory())).pack(side=tk.RIGHT)
        
        # Control buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X)
        
        def start_batch():
            if not configurations:
                messagebox.showerror("Error", "No configurations selected")
                return
            
            try:
                job_id = batch_processor.create_batch_job(
                    job_name_var.get(),
                    configurations,
                    Path(results_dir_var.get())
                )
                
                batch_processor.start_batch_job(job_id)
                messagebox.showinfo("Success", f"Batch job started: {job_id}")
                dialog.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start batch job: {e}")
        
        ttk.Button(btn_frame, text="Start Batch Job", command=start_batch).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

# Example usage and testing
def demonstrate_file_management():
    """Demonstrate file management capabilities"""
    
    print("CellQuantGUI File Management Demo")
    print("=" * 40)
    
    # Initialize file manager
    file_manager = FileManager()
    
    # Scan current directory for demo
    try:
        demo_dir = Path("demo_data")  # Would contain sample images
        if demo_dir.exists():
            files = file_manager.scan_directory(demo_dir)
            
            print(f"\nFound {len(files)} files:")
            for file_info in files[:5]:  # Show first 5
                print(f"  - {file_info.path.name} ({file_info.size_mb:.1f} MB)")
            
            # Organize files
            organized = file_manager.organize_files(files, "extension")
            print(f"\nFiles by extension:")
            for ext, file_list in organized.items():
                print(f"  {ext}: {len(file_list)} files")
            
            # Check for duplicates
            duplicates = file_manager.find_duplicate_files(files)
            if duplicates:
                print(f"\nFound {len(duplicates)} groups of duplicate files")
        else:
            print("Demo data directory not found")
    
    except Exception as e:
        print(f"Demo error: {e}")
    
    # Initialize batch processor
    batch_processor = BatchProcessor(max_workers=2)
    
    print(f"\nBatch processor initialized with {batch_processor.max_workers} workers")
    print(f"Active jobs: {len(batch_processor.list_jobs())}")

if __name__ == "__main__":
    demonstrate_file_management()