# PyLinPro - Advanced Structural Analysis System

Python/Streamlit implementation of JLinPro with extended capabilities for structural analysis and design.

## Overview

PyLinPro is a modern finite element analysis (FEA) software for structural engineering, featuring:

- **2D & 3D Frame Analysis**: Comprehensive static, modal, and dynamic analysis
- **Design Code Checking**: TCVN 5574:2018, ACI 318-25, Eurocode 2
- **Structural Optimization**: Genetic algorithm-based section optimization
- **Interactive Visualization**: 3D plotting with Plotly
- **Web-Based UI**: Streamlit interface, no installation required

## Project Structure

```
jlinpro-py/
├── src/
│   ├── core/           # FEM engine, solvers, data structures
│   ├── elements/       # Element library (Beam2D, Beam3D, Truss2D, Truss3D)
│   ├── design/         # Code checking modules (TCVN, ACI, EC2)
│   ├── optimization/   # GA optimization engine
│   └── utils/          # Helper functions, validators
├── app/
│   ├── pages/          # Streamlit multipage structure
│   ├── components/     # Reusable UI components
│   └── visualization/  # Plotly 3D plotting modules
├── tests/              # Unit and integration tests
├── data/               # Example models and design standards
└── docs/               # Documentation
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup

```bash
# Clone or navigate to project directory
cd /home/hha/work/jlinpro-py

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

```bash
# Launch Streamlit application
streamlit run app/main.py
```

The application will open in your default browser at `http://localhost:8501`

## Features

### Analysis Capabilities
- **Static Analysis**: Linear elastic analysis for 2D/3D frames and trusses
- **Modal Analysis**: Natural frequencies and mode shapes
- **Dynamic Analysis**: Time-history response using Duhamel integration
- **Support for**: Distributed loads, concentrated loads, thermal effects

### Design Modules
- **TCVN 5574:2018**: Vietnamese concrete design standard
- **ACI 318-25**: American Concrete Institute building code
- **Eurocode 2**: European concrete design standard

### Optimization
- Genetic algorithm-based section optimization
- Multi-objective optimization (weight vs. performance)
- Design code constraint enforcement

## Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Format code
black src/ app/ tests/

# Type checking
mypy src/ app/

# Linting
pylint src/ app/
```

## Docker Deployment

```bash
# Build image
docker build -t pylinpro .

# Run container
docker run -p 8501:8501 pylinpro
```

## Documentation

See `docs/` directory for:
- User Guide
- API Reference
- Developer Guide
- Migration Notes from JLinPro

## Migration Status

This project is migrated from [JLinPro](https://sourceforge.net/projects/jlinpro/) (Java) to Python/Streamlit.

**Current Phase**: Foundation & Core Engine Migration

See `MIGRATION_PROMPTS.md` for detailed migration roadmap.

## License

MIT License - See LICENSE file for details

## Contact

- Email: ha.nguyen@hydrostructai.com
- Website: https://hydrostructai.github.io

## Acknowledgments

Original JLinPro developed by Enes Siljak. This Python implementation extends the original concept with modern web technologies and additional design/optimization capabilities.
