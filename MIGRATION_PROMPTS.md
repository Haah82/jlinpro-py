---
aliases:
  - JLinPro Python Migration Prompts
created: 2026-01-28
progress: planning
tags:
  - Prompt Engineering
  - Python
  - Streamlit
  - Structure
  - FEM
category:
  - Development
  - Migration
---

# JLinPro to Python/Streamlit Migration: Agent Execution Prompts

Sequential prompts for AI Agent to migrate JLinPro (Java) to Python with Streamlit UI, including structural design code checking (TCVN 5574:2018, ACI 318-25, Eurocode 2) and genetic algorithm optimization.

---

## Phase 1: Foundation & Core Engine Migration

### Prompt 1.1: Project Initialization & Structure Setup
```
Initialize new Python project at `/home/hha/work/jlinpro-py/` following Clean Architecture pattern for Streamlit applications.

Directory structure:
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
├── tests/
│   ├── unit/
│   └── integration/
├── data/
│   ├── examples/       # Sample models (.json, .csv)
│   └── standards/      # Design code parameters
├── docs/
├── requirements.txt
├── setup.py
├── Dockerfile
└── README.md

Create `requirements.txt`:
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
streamlit>=1.28.0
plotly>=5.17.0
matplotlib>=3.7.0
deap>=1.4.0
pydantic>=2.0.0
pytest>=7.4.0
black>=23.0.0
```

### Prompt 1.2: Core Data Structures (Node, Material, Section)
```
Reference source: `/home/hha/work/jlinpro/jlinpro/src/structure/Node.java` and related classes.

Implement in `src/core/structures.py` using Pydantic BaseModel:

1. **Node** class:
   - Attributes: id (int), x (float), y (float), z (float), restraints (list[bool] of length 6 for 3D)
   - Methods: get_dofs(), is_restrained(dof_index), set_restraint(dof_index, value)
   - Support both 2D (3 DOFs) and 3D (6 DOFs) with automatic detection

2. **Material** class:
   - Attributes: name (str), E (float), G (float), nu (float), rho (float)
   - Factory methods: from_steel(), from_concrete(fc), from_timber()
   - Validation: E > 0, 0 < nu < 0.5

3. **Section** class (abstract base):
   - Subclasses: RectangularSection, CircularSection, ISection, CustomSection
   - Properties: A, Ix, Iy, Iz, J (computed from geometry)
   - Methods: get_properties(), validate_geometry()

Include comprehensive type hints and Google-style docstrings with examples.
```

### Prompt 1.3: Finite Element Library - 2D Elements
```
Reference: `/home/hha/work/jlinpro/jlinpro/src/elements/Beam2D.java` and `Truss2D.java`

Implement in `src/elements/`:

1. **AbstractElement** (base class in `base.py`):
   - Abstract methods: get_stiffness_local(), get_transformation_matrix(), get_dof_indices()
   - Common methods: get_stiffness_global(), get_length(), validate_connectivity()

2. **Truss2D** in `truss2d.py`:
   - Local stiffness matrix (2x2): k_local = (E*A/L) * [[1, -1], [-1, 1]]
   - Transformation matrix: T (2D rotation based on element orientation)
   - Global stiffness: k_global = T^T * k_local * T

3. **Beam2D** in `beam2d.py`:
   - Local stiffness matrix (6x6): Include axial, shear, bending stiffness
   - Release conditions: bit flags for moment/shear/axial releases at nodes
   - Methods: get_internal_forces(u_global), get_displacement_local(u_global)

Unit tests in `tests/unit/test_elements_2d.py`:
- Verify cantilever beam tip deflection: δ = PL³/(3EI)
- Verify simple truss: compare with hand calculation
```

### Prompt 1.4: Assembly & Linear Solver
```
Implement in `src/core/structure.py`:

**Structure** class:
- Attributes: nodes (dict), elements (dict), loads (dict), analysis_results (dict)
- Methods:

1. `assemble_global_stiffness()`:
   - Use scipy.sparse.lil_matrix for efficient assembly
   - Loop through elements, get global k, scatter to K_global based on DOF mapping
   - Return: K_global as scipy.sparse.csr_matrix

2. `apply_boundary_conditions(K, F)`:
   - Method: Penalty method (multiply diagonal by 1e12 for restrained DOFs)
   - Alternative: Matrix reduction (remove rows/columns for restrained DOFs)
   - Return: K_modified, F_modified, dof_map

3. `solve_static()`:
   - Solve: K_modified * U = F_modified using scipy.sparse.linalg.spsolve
   - Post-process: reactions = K_original @ U - F_applied
   - Calculate element internal forces: call element.get_internal_forces(U)

4. `get_results_summary()`:
   - Return pandas DataFrame with: node displacements, reactions, element forces

Validation test: Simply-supported beam with uniform load, compare with beam theory.
```

---

## Phase 2: Streamlit UI & Visualization

### Prompt 2.1: Streamlit App Structure
```
Create `app/main.py`:

Page configuration:
st.set_page_config(page_title="PyLinPro", layout="wide", initial_sidebar_state="expanded")

Sidebar navigation:
- Radio buttons: ["🏗️ Modeling", "🔬 Analysis", "📊 Results", "✅ Design Check", "🧬 Optimization"]
- Store active page in st.session_state

Session state initialization:
if 'structure' not in st.session_state:
    st.session_state.structure = Structure()
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

Create pages:
- `app/pages/modeling.py`: Input nodes, elements, materials via forms and file upload
- `app/pages/analysis.py`: Run static/modal analysis, set parameters
- `app/pages/results.py`: Display tables, diagrams, deformed shape
- `app/pages/design.py`: Code checking interface
- `app/pages/optimization.py`: GA parameter setup and execution
```

### Prompt 2.2: Interactive 3D Visualization
```
Implement in `app/visualization/plotter.py`:

Use plotly.graph_objects for all 3D plots.

1. **plot_structure_3d(structure, show_labels=True)**:
   - Nodes: go.Scatter3d with mode='markers+text'
   - Elements: go.Scatter3d with mode='lines', color by element type
   - Supports: Different marker for fixed/pinned/free nodes
   - Return: plotly Figure object

2. **plot_deformed_shape(structure, scale=1.0, overlay_original=True)**:
   - Original structure: semi-transparent gray lines
   - Deformed structure: solid colored lines, displacements scaled by factor
   - Add colorbar showing displacement magnitude
   - Animation slider for scale factor

3. **plot_internal_forces(structure, force_type='M', scale=1.0)**:
   - force_type: 'N' (axial), 'V' (shear), 'M' (moment)
   - Draw force diagram perpendicular to element axis
   - Color gradient based on force magnitude
   - Max/min annotations

4. **plot_mode_shape(structure, mode_number, animate=True)**:
   - Animate mode shape oscillation if animate=True
   - Display frequency and period in title

Integrate with Streamlit using st.plotly_chart(fig, use_container_width=True)
```

---

## Phase 3: Design Code Checking Modules

### Prompt 3.1: TCVN 5574:2018 Implementation
```
Create `src/design/tcvn5574.py`:

**Reference**: TCVN 5574:2018 - Kết cấu bê tông cốt thép

Implement classes:

1. **ConcreteGrade** (dataclass):
   - Attributes: name (str), Rb (MPa), Rbt (MPa), Eb (MPa)
   - Predefined: B15, B20, B25, B30, B35, B40

2. **SteelGrade** (dataclass):
   - Attributes: name (str), Rs (MPa), Rsc (MPa), Es (MPa)
   - Predefined: CB300V, CB400V, CB500V

3. **BeamChecker**:
   - Method: `check_flexure(M, b, h, concrete, steel, As, cover=25)`
     * Calculate Mu = Rs * As * (h0 - 0.5*x) where x = Rs*As/(Rb*b)
     * Utilization ratio: UR = M / Mu
     * Return: {"status": "PASS"|"FAIL", "UR": float, "Mu": float}

4. **ColumnChecker**:
   - Method: `check_compression(N, M, b, h, concrete, steel, As_total)`
     * Consider slenderness ratio if lambda > 14
     * Check interaction diagram: (N/Nu) + (M/Mu) <= 1
     * Return: {"status": "PASS"|"FAIL", "UR": float, "Nu": float, "Mu": float}

Streamlit interface in `app/pages/design.py`:
- Dropdown: Select concrete grade (B15-B40)
- Dropdown: Select steel grade (CB300V-CB500V)
- Input: Reinforcement details (As, cover, stirrups)
- Button: "Run TCVN Check" -> iterate all beam/column elements
- Display: Results table with color coding (green=PASS, red=FAIL)
```

### Prompt 3.2: ACI 318-25 & Eurocode 2 Modules
```
Create `src/design/aci318.py`:

**Reference**: ACI 318-25 Building Code Requirements for Structural Concrete

1. **ConcreteACI** (dataclass):
   - fc_prime (psi), lambda_factor (1.0 for normal weight)
   - Methods: get_Ec() per ACI 19.2.2.1

2. **FlexureChecker**:
   - Method: `check_flexure(Mu, b, h, fc, fy, As, phi=0.9)`
     * Calculate a = As*fy/(0.85*fc*b)
     * Nominal strength: Mn = As*fy*(d - a/2)
     * Design strength: phi*Mn
     * Check: Mu <= phi*Mn

3. **ShearChecker**:
   - Method: `check_shear(Vu, b, d, fc, Av, s, phi=0.75)`
     * Vc = 2*sqrt(fc)*b*d (in psi units)
     * Vs = Av*fy*d/s
     * Check: Vu <= phi*(Vc + Vs)

Create `src/design/eurocode2.py`:

**Reference**: EN 1992-1-1:2004 Eurocode 2

1. **ConcreteEC2** (dataclass):
   - fck (MPa), gamma_c (1.5), alpha_cc (1.0)
   - fcd = alpha_cc * fck / gamma_c

2. **FlexureCheckerEC2**:
   - Method: `check_ULS_flexure(MEd, b, h, fck, fyk, As)`
     * Calculate x from equilibrium
     * MRd = As*fyd*(d - 0.4*x)
     * Check: MEd <= MRd

Design strategy pattern in `src/design/base.py`:
- Abstract class: DesignCode
- Implementations: TCVN5574Code, ACI318Code, Eurocode2Code
- User selects via dropdown, backend calls appropriate checker
```

### Prompt 3.3: Genetic Algorithm Optimization
```
Create `src/optimization/ga_optimizer.py`:

Use DEAP library (Distributed Evolutionary Algorithms in Python).

**Problem formulation**:
- Objective: Minimize total structural weight = sum(rho * A * L) for all elements
- Variables: Section sizes from discrete catalog (e.g., [100x100, 150x150, 200x200, ...])
- Constraints:
  1. Stress: sigma <= sigma_allowable
  2. Displacement: delta <= delta_allowable (e.g., L/250)
  3. Code compliance: All elements pass TCVN/ACI/EC2 check

**Implementation**:

1. **Genome encoding**:
   - Individual: list of integers, each representing section index for one element
   - Example: [0, 2, 1, 3] -> element_0 uses section_catalog[0], etc.

2. **Fitness function**:
   ```python
   def evaluate(individual, structure, section_catalog):
       # Assign sections from individual
       for i, elem in enumerate(structure.elements):
           elem.section = section_catalog[individual[i]]
       
       # Run analysis
       structure.solve_static()
       
       # Calculate weight
       weight = sum(elem.get_weight() for elem in structure.elements.values())
       
       # Calculate penalty for constraint violation
       penalty = 0
       for elem in structure.elements.values():
           if elem.max_stress > elem.allowable_stress:
               penalty += 1e6
       
       for node in structure.nodes.values():
           if abs(node.displacement_y) > structure.max_displacement:
               penalty += 1e6
       
       return (weight + penalty,)  # DEAP expects tuple
   ```

3. **GA setup**:
   ```python
   from deap import base, creator, tools, algorithms
   
   creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
   creator.create("Individual", list, fitness=creator.FitnessMin)
   
   toolbox = base.Toolbox()
   toolbox.register("attr_int", random.randint, 0, len(section_catalog)-1)
   toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=n_elements)
   toolbox.register("population", tools.initRepeat, list, toolbox.individual)
   toolbox.register("evaluate", evaluate_function)
   toolbox.register("mate", tools.cxTwoPoint)
   toolbox.register("mutate", tools.mutUniformInt, low=0, up=len(section_catalog)-1, indpb=0.2)
   toolbox.register("select", tools.selTournament, tournsize=3)
   ```

4. **Streamlit interface** (`app/pages/optimization.py`):
   - Inputs: Population size, generations, mutation rate, crossover rate
   - Button: "Start Optimization"
   - Real-time plot: Best fitness vs generation (using st.line_chart, update with st.empty())
   - Results: Optimal section assignment, weight reduction percentage
```

---

## Phase 4: 3D Extension & Advanced Features

### Prompt 4.1: 3D Frame Element Implementation
```
Create `src/elements/beam3d.py`:

**Beam3D** class for space frame analysis (6 DOFs per node).

1. **Local stiffness matrix (12x12)**:
   - Axial: EA/L
   - Torsion: GJ/L
   - Bending Y: 4EIz/L, 2EIz/L (strong axis)
   - Bending Z: 4EIy/L, 2EIy/L (weak axis)
   - Assemble into 12x12 symmetric matrix

2. **3D transformation matrix (12x12)**:
   - Calculate direction cosines: cx, cy, cz from node coordinates
   - Local axes: x_local along element, y_local and z_local perpendicular
   - Handle arbitrary orientation using rotation vector or Euler angles
   - Reference: "Matrix Analysis of Structures" by Kassimali

3. **Global stiffness**: K_global = T^T * K_local * T

4. **Backward compatibility**:
   - Detect 2D problem: if all z-coordinates == 0 and no z-direction loads
   - Automatically reduce to 2D analysis (3 DOFs) for computational efficiency

Unit test: 3D cantilever with tip load, verify deflection formula.

Implement **Truss3D** similarly in `src/elements/truss3d.py`.
```

### Prompt 4.2: Modal Analysis & Dynamic Response
```
Reference: `/home/hha/work/jlinpro/jlinpro/src/structure/Structure.java` (lines 401-600)

Implement in `src/core/dynamic.py`:

1. **Mass matrix assembly**:
   - Lumped mass: Diagonal matrix with nodal masses
   - Consistent mass: Integrate shape functions (6x6 for beam element)
   - Method: `assemble_mass_matrix(mass_type='lumped')`

2. **Eigenvalue problem solver**:
   ```python
   from scipy.sparse.linalg import eigsh
   
   def solve_modal(K, M, n_modes=10):
       # Solve: (K - omega^2 * M) * phi = 0
       # Using shift-invert mode for better convergence
       eigenvalues, eigenvectors = eigsh(K, k=n_modes, M=M, sigma=0, which='LM')
       
       frequencies = np.sqrt(eigenvalues) / (2 * np.pi)  # Hz
       periods = 1.0 / frequencies  # seconds
       
       return {
           'frequencies': frequencies,
           'periods': periods,
           'mode_shapes': eigenvectors
       }
   ```

3. **Time-history analysis** (Duhamel integral method):
   - Reference JLinPro implementation (lines 478-556)
   - Modal superposition: u(t) = sum(phi_i * q_i(t))
   - Integrate using Newmark-beta or Duhamel convolution
   - Support seismic input (acceleration time history)

4. **Streamlit visualization**:
   - Display modal frequencies table
   - Animate mode shapes (plotly frames)
   - Plot response time history for selected node
```

### Prompt 4.3: Documentation & Deployment
```
1. **User Guide** (`docs/USER_GUIDE.md`):
   - Getting Started: Installation, launching app
   - Modeling: Creating nodes, elements, applying loads
   - Analysis: Running static/modal/dynamic analysis
   - Design Checks: Using TCVN/ACI/EC2 modules
   - Optimization: Setting up GA runs
   - Examples: Step-by-step tutorials for 5 benchmark problems

2. **API Documentation** (`docs/API_REFERENCE.md`):
   - Auto-generate using Sphinx or pdoc
   - Document all public classes and methods
   - Include code examples for programmatic usage

3. **Dockerfile**:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

4. **Code quality**:
   - Run black formatter: `black src/ app/ tests/`
   - Run pylint: `pylint src/ app/`
   - Type checking: `mypy src/ app/`
   - Test coverage: `pytest --cov=src tests/`

5. **CI/CD** (`.github/workflows/ci.yml`):
   - Automated testing on push
   - Code quality checks
   - Build and push Docker image

6. **Performance optimization**:
   - Profile critical functions using cProfile
   - Optimize sparse matrix operations
   - Implement result caching with st.cache_data
   - Add progress bars for long computations
```

---

## Execution Guidelines

**Sequential execution**: Execute prompts in order 1.1 → 1.2 → ... → 4.3

**Validation after each prompt**:
- Run unit tests
- Verify output matches specification
- Check code quality (linting, type hints)

**Version control**: Git commit after each completed prompt with descriptive message

**Testing strategy**:
- Unit tests: Individual functions and classes
- Integration tests: Complete analysis workflows
- Validation tests: Compare with published solutions or commercial software

**Code standards**:
- PEP 8 compliance
- Type hints on all functions
- Google-style docstrings
- Maximum function length: 50 lines
- Maximum file length: 500 lines
