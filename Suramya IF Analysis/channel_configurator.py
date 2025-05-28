#!/usr/bin/env python3
"""
Fixed Channel Configurator - Properly groups files by channel suffix
Only shows unique channels (C0, C1, C2, C3) instead of every single file
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import re
from typing import List, Dict, Optional
from PIL import Image
import logging

class ChannelConfigurator:
    """Dialog for configuring channel information and assignments - FIXED VERSION"""
    
    def __init__(self, parent, condition_name: str, image_directory: str):
        self.parent = parent
        self.condition_name = condition_name
        self.image_directory = Path(image_directory)
        self.channels = []
        self.result = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(f"Configure Channels - {condition_name}")
        self.dialog.geometry("1200x800")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (1200 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (800 // 2)
        self.dialog.geometry(f"1200x800+{x}+{y}")
        
        self.setup_dialog()
        self.auto_detect_channels()
    
    def setup_dialog(self):
        """Setup the dialog interface"""
        
        # Header
        header_frame = ttk.Frame(self.dialog)
        header_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(header_frame, 
                 text=f"Configure channels for condition: {self.condition_name}",
                 font=("Arial", 14, "bold")).pack()
        
        ttk.Label(header_frame, 
                 text="Assign channel types and specify which channels to quantify.",
                 font=("Arial", 10)).pack()
        
        # Main content frame with scrollable area
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create scrollable frame for channel configuration
        self.setup_scrollable_frame(main_frame)
        
        # Preview frame
        preview_frame = ttk.LabelFrame(main_frame, text="Channel Preview")
        preview_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.preview_label = ttk.Label(preview_frame, text="Select a channel to preview")
        self.preview_label.pack(pady=10)
        
        # Template buttons
        template_frame = ttk.Frame(self.dialog)
        template_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(template_frame, text="Load Template", 
                  command=self.load_template).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(template_frame, text="Save Template", 
                  command=self.save_template).pack(side=tk.LEFT)
        
        # Bottom buttons
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Cancel", 
                  command=self.cancel).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="OK", 
                  command=self.accept).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="Auto-Detect", 
                  command=self.auto_detect_channels).pack(side=tk.LEFT)
    
    def setup_scrollable_frame(self, parent):
        """Create scrollable frame for channel configuration"""
        
        # Create canvas and scrollbar
        canvas = tk.Canvas(parent, height=400)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Headers
        headers_frame = ttk.Frame(self.scrollable_frame)
        headers_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(headers_frame, text="Channel", width=15).grid(row=0, column=0, padx=2)
        ttk.Label(headers_frame, text="Type", width=12).grid(row=0, column=1, padx=2)
        ttk.Label(headers_frame, text="Purpose", width=15).grid(row=0, column=2, padx=2)
        ttk.Label(headers_frame, text="Quantify", width=8).grid(row=0, column=3, padx=2)
        ttk.Label(headers_frame, text="Nuclear Only", width=10).grid(row=0, column=4, padx=2)
        ttk.Label(headers_frame, text="Wavelength", width=12).grid(row=0, column=5, padx=2)
        ttk.Label(headers_frame, text="Preview", width=8).grid(row=0, column=6, padx=2)
        
        # Buttons frame
        button_controls = ttk.Frame(self.scrollable_frame)
        button_controls.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_controls, text="Add Channel", 
                  command=self.add_channel).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_controls, text="Remove Selected", 
                  command=self.remove_channel).pack(side=tk.LEFT)
        
        # Content frame for channel rows
        self.content_frame = ttk.Frame(self.scrollable_frame)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=5)
    
    def auto_detect_channels(self):
        """FIXED: Automatically detect and configure channels based on filenames"""
        
        self.logger.info(f"Auto-detecting channels in: {self.image_directory}")
        
        # Clear existing channels
        self.channels = []
        
        # Get all image files
        image_extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
        all_images = []
        
        for ext in image_extensions:
            all_images.extend(self.image_directory.glob(f'*{ext}'))
            all_images.extend(self.image_directory.glob(f'*{ext.upper()}'))
        
        if not all_images:
            messagebox.showwarning("Warning", "No image files found in directory")
            return
        
        self.logger.info(f"Found {len(all_images)} total image files")
        
        # Group files by channel suffix and detect unique channels
        channel_groups = self.group_files_by_channel(all_images)
        
        self.logger.info(f"Detected {len(channel_groups)} unique channels: {list(channel_groups.keys())}")
        
        # Create channel configurations
        for channel_suffix, file_list in channel_groups.items():
            
            # Use the first file as representative
            representative_file = file_list[0]
            
            channel_config = {
                'name': self.generate_channel_name(channel_suffix),
                'suffix': channel_suffix,
                'files': file_list,
                'representative_file': representative_file,
                'type': self.guess_channel_type(channel_suffix),
                'purpose': self.guess_channel_purpose(channel_suffix),
                'quantify': self.should_quantify_by_default(channel_suffix),
                'nuclear_only': self.is_likely_nuclear(channel_suffix),
                'wavelength': self.guess_wavelength(channel_suffix),
                'source': 'single_file'
            }
            
            self.channels.append(channel_config)
        
        # Sort channels by suffix (C0, C1, C2, C3...)
        self.channels.sort(key=lambda x: x['suffix'])
        
        # Refresh display
        self.refresh_channel_display()
        
        messagebox.showinfo("Success", 
                           f"Auto-detected {len(self.channels)} channels:\n" + 
                           "\n".join([f"• {ch['name']} ({ch['suffix']}) - {len(ch['files'])} files" 
                                     for ch in self.channels]))
    
    def group_files_by_channel(self, image_files: List[Path]) -> Dict[str, List[Path]]:
        """Group image files by their channel suffix (C0, C1, C2, etc.)"""
        
        channel_groups = {}
        
        for file_path in image_files:
            filename = file_path.name
            
            # Extract channel suffix using various patterns
            channel_suffix = self.extract_channel_suffix(filename)
            
            if channel_suffix:
                if channel_suffix not in channel_groups:
                    channel_groups[channel_suffix] = []
                channel_groups[channel_suffix].append(file_path)
            else:
                # Files without channel suffix go to "Unknown" group
                if 'Unknown' not in channel_groups:
                    channel_groups['Unknown'] = []
                channel_groups['Unknown'].append(file_path)
        
        return channel_groups
    
    def extract_channel_suffix(self, filename: str) -> Optional[str]:
        """Extract channel suffix from filename"""
        
        # Common channel patterns
        patterns = [
            r'[_\-]C(\d+)\.tiff?$',        # _C0.tif, _C1.tiff
            r'[_\-]c(\d+)\.tiff?$',        # _c0.tif, _c1.tiff  
            r'[_\-]CH(\d+)\.tiff?$',       # _CH0.tif, _CH1.tiff
            r'[_\-]ch(\d+)\.tiff?$',       # _ch0.tif, _ch1.tiff
            r'[_\-]channel(\d+)\.tiff?$',  # _channel0.tif
            r'[_\-]Channel(\d+)\.tiff?$',  # _Channel0.tif
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                channel_num = match.group(1)
                return f"C{channel_num}"
        
        return None
    
    def generate_channel_name(self, channel_suffix: str) -> str:
        """Generate a descriptive channel name"""
        
        if channel_suffix == 'C0':
            return 'DAPI_Nuclear'
        elif channel_suffix == 'C1':
            return 'Channel_1'
        elif channel_suffix == 'C2':
            return 'Channel_2'
        elif channel_suffix == 'C3':
            return 'Channel_3'
        elif channel_suffix == 'Unknown':
            return 'Unknown_Channel'
        else:
            return f'Channel_{channel_suffix}'
    
    def guess_channel_type(self, channel_suffix: str) -> str:
        """Guess channel type based on suffix and naming"""
        
        if channel_suffix == 'C0':
            return 'nuclear'  # Usually DAPI
        else:
            return 'cellular'
    
    def guess_channel_purpose(self, channel_suffix: str) -> str:
        """Guess channel purpose"""
        
        if channel_suffix == 'C0':
            return 'segmentation'  # DAPI for segmentation
        else:
            return 'quantification'
    
    def should_quantify_by_default(self, channel_suffix: str) -> bool:
        """Determine if channel should be quantified by default"""
        
        # Don't quantify DAPI by default, quantify others
        return channel_suffix != 'C0'
    
    def is_likely_nuclear(self, channel_suffix: str) -> bool:
        """Determine if signal is likely nuclear-only"""
        
        return channel_suffix == 'C0'  # DAPI is nuclear
    
    def guess_wavelength(self, channel_suffix: str) -> str:
        """Guess wavelength based on channel"""
        
        wavelength_map = {
            'C0': '405nm',  # DAPI
            'C1': '488nm',  # Common green
            'C2': '568nm',  # Common red
            'C3': '647nm',  # Common far-red
        }
        
        return wavelength_map.get(channel_suffix, '')
    
    def refresh_channel_display(self):
        """Refresh the channel configuration display"""
        
        # Clear existing widgets
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Create rows for each channel
        for i, channel in enumerate(self.channels):
            self.create_channel_row(i, channel)
    
    def create_channel_row(self, row_idx: int, channel: Dict):
        """Create widget row for a single channel"""
        
        row_frame = ttk.Frame(self.content_frame)
        row_frame.pack(fill=tk.X, pady=2)
        
        # Channel name (editable)
        name_var = tk.StringVar(value=channel['name'])
        name_entry = ttk.Entry(row_frame, textvariable=name_var, width=15)
        name_entry.grid(row=0, column=0, padx=2, sticky="ew")
        channel['name_var'] = name_var
        
        # Type dropdown
        type_var = tk.StringVar(value=channel['type'])
        type_combo = ttk.Combobox(row_frame, textvariable=type_var, 
                                 values=['nuclear', 'cellular', 'organelle'], 
                                 width=10, state="readonly")
        type_combo.grid(row=0, column=1, padx=2)
        channel['type_var'] = type_var
        
        # Purpose dropdown
        purpose_var = tk.StringVar(value=channel['purpose'])
        purpose_combo = ttk.Combobox(row_frame, textvariable=purpose_var,
                                   values=['segmentation', 'quantification', 'reference'],
                                   width=13, state="readonly")
        purpose_combo.grid(row=0, column=2, padx=2)
        channel['purpose_var'] = purpose_var
        
        # Quantify checkbox
        quantify_var = tk.BooleanVar(value=channel['quantify'])
        quantify_check = ttk.Checkbutton(row_frame, variable=quantify_var)
        quantify_check.grid(row=0, column=3, padx=2)
        channel['quantify_var'] = quantify_var
        
        # Nuclear only checkbox
        nuclear_var = tk.BooleanVar(value=channel['nuclear_only'])
        nuclear_check = ttk.Checkbutton(row_frame, variable=nuclear_var)
        nuclear_check.grid(row=0, column=4, padx=2)
        channel['nuclear_only_var'] = nuclear_var
        
        # Wavelength entry
        wavelength_var = tk.StringVar(value=channel['wavelength'])
        wavelength_entry = ttk.Entry(row_frame, textvariable=wavelength_var, width=10)
        wavelength_entry.grid(row=0, column=5, padx=2)
        channel['wavelength_var'] = wavelength_var
        
        # Preview button
        preview_btn = ttk.Button(row_frame, text="Preview", 
                               command=lambda idx=row_idx: self.preview_channel(idx))
        preview_btn.grid(row=0, column=6, padx=2)
        
        # File count label
        file_count_label = ttk.Label(row_frame, 
                                   text=f"({len(channel['files'])} files)",
                                   font=("Arial", 8))
        file_count_label.grid(row=0, column=7, padx=2)
    
    def add_channel(self):
        """Add a new channel configuration"""
        new_channel = {
            'name': f'Channel_{len(self.channels) + 1}',
            'suffix': f'C{len(self.channels)}',
            'files': [],
            'type': 'cellular',
            'purpose': 'quantification',
            'quantify': True,
            'nuclear_only': False,
            'wavelength': '',
            'source': 'manual'
        }
        
        self.channels.append(new_channel)
        self.refresh_channel_display()
    
    def remove_channel(self):
        """Remove selected channel"""
        if self.channels:
            self.channels.pop()
            self.refresh_channel_display()
    
    def preview_channel(self, channel_idx: int):
        """Preview the selected channel"""
        if channel_idx >= len(self.channels):
            return
            
        channel = self.channels[channel_idx]
        
        if not channel['files']:
            self.preview_label.config(text="No files for this channel")
            return
        
        try:
            # Show info about the channel
            rep_file = channel['representative_file']
            info_text = f"Channel: {channel['name']}\n"
            info_text += f"Files: {len(channel['files'])}\n"
            info_text += f"Example: {rep_file.name}\n"
            info_text += f"Type: {channel['type']}\n"
            info_text += f"Purpose: {channel['purpose']}"
            
            self.preview_label.config(text=info_text)
            
        except Exception as e:
            self.preview_label.config(text=f"Preview error: {e}")
    
    def load_template(self):
        """Load channel configuration template"""
        messagebox.showinfo("Info", "Template loading will be implemented")
    
    def save_template(self):
        """Save current configuration as template"""
        messagebox.showinfo("Info", "Template saving will be implemented")
    
    def collect_channel_data(self) -> List[Dict]:
        """Collect channel data from UI widgets"""
        
        collected_channels = []
        
        for channel in self.channels:
            if 'name_var' not in channel:
                continue
                
            channel_data = {
                'name': channel['name_var'].get(),
                'type': channel['type_var'].get(),
                'purpose': channel['purpose_var'].get(),
                'quantify': channel['quantify_var'].get(),
                'nuclear_only': channel['nuclear_only_var'].get(),
                'wavelength': channel['wavelength_var'].get(),
                'files': channel['files'],
                'suffix': channel['suffix'],
                'source': channel.get('source', 'single_file')
            }
            
            collected_channels.append(channel_data)
        
        return collected_channels
    
    def validate_configuration(self) -> bool:
        """Validate the channel configuration"""
        
        if not self.channels:
            messagebox.showerror("Error", "No channels configured")
            return False
        
        # Check for at least one quantification channel
        quantify_channels = [ch for ch in self.channels 
                           if 'quantify_var' in ch and ch['quantify_var'].get()]
        
        if not quantify_channels:
            result = messagebox.askyesno("Warning", 
                                       "No channels selected for quantification. Continue anyway?")
            if not result:
                return False
        
        # Check for duplicate names
        names = [ch['name_var'].get() for ch in self.channels if 'name_var' in ch]
        if len(names) != len(set(names)):
            messagebox.showerror("Error", "Duplicate channel names found")
            return False
        
        return True
    
    def accept(self):
        """Accept the configuration and close dialog"""
        
        if not self.validate_configuration():
            return
        
        self.result = self.collect_channel_data()
        self.dialog.destroy()
    
    def cancel(self):
        """Cancel and close dialog"""
        self.result = None
        self.dialog.destroy()
    
    def show(self) -> Optional[List[Dict]]:
        """Show the dialog and return the result"""
        
        self.dialog.wait_window()
        return self.result


class ConditionManager:
    """Manages experimental conditions and their configurations"""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.logger = logging.getLogger(__name__)
    
    def add_condition_dialog(self):
        """Show dialog to add a new condition"""
        
        # Create dialog window
        dialog = tk.Toplevel(self.parent_app.root)
        dialog.title("Add New Condition")
        dialog.geometry("600x400")
        dialog.transient(self.parent_app.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (600 // 2)
        y = (dialog.winfo_screenheight() // 2) - (400 // 2)
        dialog.geometry(f"600x400+{x}+{y}")
        
        # Variables
        condition_name_var = tk.StringVar(value="")
        condition_dir_var = tk.StringVar(value="")
        description_var = tk.StringVar(value="")
        
        # Main frame
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        ttk.Label(main_frame, text="Add New Experimental Condition", 
                 font=("Arial", 14, "bold")).pack(pady=(0, 20))
        
        # Condition name
        name_frame = ttk.Frame(main_frame)
        name_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(name_frame, text="Condition Name:", width=15).pack(side=tk.LEFT)
        ttk.Entry(name_frame, textvariable=condition_name_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Directory selection
        dir_frame = ttk.Frame(main_frame)
        dir_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(dir_frame, text="Image Directory:", width=15).pack(side=tk.LEFT)
        ttk.Entry(dir_frame, textvariable=condition_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        ttk.Button(dir_frame, text="Browse", 
                  command=lambda: self.browse_directory(condition_dir_var)).pack(side=tk.RIGHT)
        
        # Description
        desc_frame = ttk.Frame(main_frame)
        desc_frame.pack(fill=tk.X, pady=(0, 20))
        ttk.Label(desc_frame, text="Description:", width=15).pack(side=tk.LEFT, anchor=tk.N)
        desc_text = tk.Text(desc_frame, height=4, wrap=tk.WORD)
        desc_text.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Image list frame
        list_frame = ttk.LabelFrame(main_frame, text="Images Found")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Create listbox with scrollbar
        listbox_frame = ttk.Frame(list_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        images_listbox = tk.Listbox(listbox_frame)
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=images_listbox.yview)
        images_listbox.configure(yscrollcommand=scrollbar.set)
        
        images_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Update images list when directory changes
        def update_images_list(*args):
            directory = condition_dir_var.get()
            if directory and Path(directory).exists():
                images = self.get_images_from_directory(Path(directory))
                images_listbox.delete(0, tk.END)
                for img in images:
                    images_listbox.insert(tk.END, img.name)
                
                # Group by channels and show summary
                channel_groups = self.group_images_by_channel(images)
                summary_text = f"Found {len(images)} images in {len(channel_groups)} channel groups"
                list_frame.config(text=f"Images Found - {summary_text}")
        
        condition_dir_var.trace('w', update_images_list)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        def cancel_condition():
            dialog.destroy()
        
        def create_condition():
            # Validate input
            name = condition_name_var.get().strip()
            directory = condition_dir_var.get().strip()
            
            if not name:
                messagebox.showerror("Error", "Please enter a condition name")
                return
            
            if not directory or not Path(directory).exists():
                messagebox.showerror("Error", "Please select a valid directory")
                return
            
            # Get images from directory
            images = self.get_images_from_directory(Path(directory))
            if not images:
                messagebox.showerror("Error", "No supported image files found in directory")
                return
            
            # Open channel configurator
            dialog.destroy()
            
            configurator = ChannelConfigurator(self.parent_app.root, name, directory)
            channel_config = configurator.show()
            
            if channel_config:
                # Create condition data
                condition_data = {
                    'name': name,
                    'directory': directory,
                    'description': desc_text.get("1.0", tk.END).strip(),
                    'channels': channel_config
                }
                
                # Add to parent application
                self.parent_app.add_condition_to_experiment(condition_data)
        
        ttk.Button(button_frame, text="Cancel", command=cancel_condition).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Configure Channels", command=create_condition).pack(side=tk.RIGHT)
    
    def browse_directory(self, dir_var):
        """Browse for directory"""
        directory = filedialog.askdirectory(title="Select Image Directory")
        if directory:
            dir_var.set(directory)
    
    def get_images_from_directory(self, directory: Path) -> List[Path]:
        """Get list of supported images from directory"""
        
        image_extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
        images = []
        
        for ext in image_extensions:
            images.extend(directory.glob(f'*{ext}'))
            images.extend(directory.glob(f'*{ext.upper()}'))
        
        return sorted(images)
    
    def group_images_by_channel(self, images: List[Path]) -> Dict[str, List[Path]]:
        """Group images by channel suffix"""
        
        channel_groups = {}
        
        for img in images:
            # Extract channel suffix
            filename = img.name
            
            # Look for channel patterns
            channel_match = re.search(r'[_\-][cC](\d+)\.tiff?$', filename)
            if channel_match:
                channel = f"C{channel_match.group(1)}"
            else:
                channel = "Unknown"
        
            if channel not in channel_groups:
                channel_groups[channel] = []
            channel_groups[channel].append(img)
        
        return channel_groups


# Quick test function
def test_channel_configurator():
    """Test the channel configurator with sample data"""
    
    root = tk.Tk()
    root.withdraw()  # Hide main window
    
    # Create test directory path
    test_dir = "/path/to/test/images"
    
    configurator = ChannelConfigurator(root, "Test Condition", test_dir)
    result = configurator.show()
    
    if result:
        print("Channel configuration result:")
        for i, channel in enumerate(result):
            print(f"  Channel {i+1}: {channel['name']} ({channel['type']}) - {len(channel['files'])} files")
    else:
        print("Configuration cancelled")
    
    root.destroy()


if __name__ == "__main__":
    test_channel_configurator()