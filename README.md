README.md
# CellQuantGUI - Quantitative Microscopy Analysis Platform

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Cellpose](https://img.shields.io/badge/Cellpose-4.0+-green.svg)](https://github.com/MouseLand/cellpose)

A comprehensive quantitative microscopy analysis platform designed for automated cell segmentation and fluorescence quantification. CellQuantGUI bridges the gap between complex image analysis algorithms and user-friendly interfaces, enabling biologists to perform rigorous quantitative analysis without programming expertise.

## 🎯 Key Features

- **Multi-channel image processing** - Handles TIFF, PNG, JPEG formats
- **Automated cell segmentation** - Cellpose integration with fallback algorithms
- **Quantitative analysis** - CTCF (Corrected Total Cell Fluorescence) calculations
- **Batch processing** - High-throughput analysis capabilities
- **Statistical analysis** - Built-in statistical testing and visualization
- **Quality control** - Comprehensive validation and reporting
- **Template system** - Pre-configured analysis workflows

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/CellQuantGUI.git
   cd CellQuantGUI
   ```

2. **Run the installation script:**
   ```bash
   python installation_setup.py
   ```
   
   Or manually install dependencies:
   ```bash
   pip install numpy pandas matplotlib pillow scikit-image cellpose torch torchvision
   ```

3. **Launch the application:**
   ```bash
   python launch_cellquant.py
   ```

### Quick Example

```python
from cellquant_main import MainApplication

# Launch GUI
app = MainApplication()
app.run()

# Or run batch analysis
from example_usage import run_batch_analysis
run_batch_analysis()
```

## 📁 Project Structure

```
CellQuantGUI/
├── cellquant_main.py              # Main GUI application
├── analysis_pipeline.py           # Core analysis engine
├── channel_configurator.py        # Channel setup dialogs
├── visualization_module.py        # Data visualization
├── template_manager.py           # Analysis templates
├── file_management.py            # File operations & batch processing
├── quality_control.py           # Quality assurance
├── installation_setup.py        # Installation & setup
├── example_usage.py             # Usage examples
├── launch_cellquant.py          # Application launcher
├── documentation.md             # Developer documentation
├── documentation-CellQuantGUI.md # System documentation
└── README.md                    # This file
```

## 🔬 Supported Analysis Types

- **DNA Damage Analysis** - γH2AX foci quantification
- **Mitochondrial Analysis** - Mitochondrial network quantification
- **Protein Localization** - Subcellular distribution analysis
- **Cell Cycle Analysis** - EdU/BrdU incorporation studies
- **Apoptosis Analysis** - Cleaved caspase-3 quantification
- **Custom Workflows** - User-defined analysis templates

## 📊 Example Results

The software generates:
- Individual cell measurements (CSV)
- Condition summaries with statistics
- Publication-ready plots
- Quality control reports
- Segmentation overlays

## 🛠️ Requirements

- Python 3.8+
- NumPy, Pandas, Matplotlib
- Cellpose (GPU recommended)
- tkinter (GUI)
- 8GB+ RAM recommended
- NVIDIA GPU (optional, for faster segmentation)

## 📖 Documentation

- [Complete Documentation](documentation-CellQuantGUI.md) - System overview and usage
- [Developer Guide](documentation.md) - Technical implementation details
- [Installation Guide](installation_setup.py) - Comprehensive setup instructions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Cellpose** team for excellent segmentation algorithms
- **scikit-image** community for image processing tools
- Contributors and beta testers

## 📞 Support

- Open an [issue](https://github.com/yourusername/CellQuantGUI/issues) for bug reports
- Check [documentation](documentation-CellQuantGUI.md) for usage questions
- Contact: your.email@example.com

## 🔬 Citation

If you use CellQuantGUI in your research, please cite:

```
CellQuantGUI: A Comprehensive Platform for Quantitative Microscopy Analysis
[Your Name et al., Year]
```