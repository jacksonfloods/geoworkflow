Installation
============

Prerequisites
-------------

Before installing GeoWorkflow, you'll need to set up your development environment with the following software:

System Requirements
~~~~~~~~~~~~~~~~~~~

* **Operating System**: Windows 10+, macOS 10.14+, or Linux
* **RAM**: Minimum 8GB (16GB recommended for large datasets)
* **Disk Space**: At least 5GB free space for software and dependencies
* **Python**: Version 3.8 or higher (3.10 recommended)

Required Software
~~~~~~~~~~~~~~~~~

1. **Python 3.10**

   * **Download**: https://www.python.org/downloads/
   * **Installation**:
   
     - **Windows**: Download the installer and check "Add Python to PATH" during installation
     - **macOS**: Download the installer or use Homebrew: ``brew install python@3.10``
     - **Linux**: Use your package manager: ``sudo apt-get install python3.10`` (Ubuntu/Debian)

   * **Verify installation**:

     .. code-block:: bash

        python --version  # Should show Python 3.8 or higher

2. **Conda (Anaconda or Miniconda)**

   We **strongly recommend** using Conda for dependency management, as it handles complex geospatial libraries (GDAL, GEOS, PROJ) much better than pip alone.

   **Option A: Miniconda (Recommended - Smaller Download)**
   
   * **Download**: https://docs.conda.io/en/latest/miniconda.html
   * **Size**: ~50MB installer
   * **What it includes**: Minimal conda installation + Python
   
   **Option B: Anaconda (Full Distribution)**
   
   * **Download**: https://www.anaconda.com/download
   * **Size**: ~500MB installer  
   * **What it includes**: Conda + Python + 250+ pre-installed packages + Anaconda Navigator GUI

   **Installation Steps**:

   **Windows**:

   1. Download the Windows installer (.exe)
   2. Run the installer
   3. Accept defaults (recommended to add to PATH when prompted)
   4. Restart your terminal/command prompt

   **macOS**:

   1. Download the macOS installer (.pkg or .sh)
   2. For .pkg: Double-click and follow prompts
   3. For .sh: Run ``bash Miniconda3-latest-MacOSX-x86_64.sh``
   4. Restart your terminal

   **Linux**:

   1. Download the Linux installer (.sh)
   2. Run: ``bash Miniconda3-latest-Linux-x86_64.sh``
   3. Follow the prompts
   4. Restart your terminal or run ``source ~/.bashrc``

   **Verify Conda Installation**:

   .. code-block:: bash

      conda --version  # Should show conda 4.x.x or higher
      conda info       # Shows detailed conda information

3. **Git (for cloning the repository)**

   * **Download**: https://git-scm.com/downloads
   * **Windows**: Use Git for Windows installer
   * **macOS**: Included with Xcode Command Line Tools or use Homebrew: ``brew install git``
   * **Linux**: ``sudo apt-get install git`` (Ubuntu/Debian)

   **Verify Git Installation**:

   .. code-block:: bash

      git --version  # Should show git version 2.x.x or higher

Optional but Recommended
~~~~~~~~~~~~~~~~~~~~~~~~

* **Visual Studio Code**: https://code.visualstudio.com/ - Excellent Python IDE with Jupyter support
* **Git GUI Client**: GitKraken, GitHub Desktop, or SourceTree for easier Git operations

Installation Steps
------------------

Automated Installation (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide an automated installation script that handles all the steps below:

.. code-block:: bash

   # Download and run the installer
   curl -fsSL https://raw.githubusercontent.com/jacksonfloods/geoworkflow/main/install.sh -o install.sh
   chmod +x install.sh
   ./install.sh

The script will:

* Install Miniconda (if not present)
* Install Git (if not present)
* Clone the repository
* Create the conda environment
* Install GeoWorkflow package
* Verify the installation

Skip to :ref:`verify-installation` if using the automated installer.

Manual Installation Using Conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the preferred installation method as it handles all geospatial dependencies automatically.

**Step 1: Clone the Repository**

.. code-block:: bash

   # Navigate to where you want to install
   cd ~/Projects  # or C:\Users\YourName\Projects on Windows

   # Clone the repository
   git clone https://github.com/jacksonfloods/geoworkflow.git
   cd geoworkflow

**Step 2: Create Conda Environment**

This will install Python 3.10 and all required geospatial libraries (GDAL, GEOS, PROJ, etc.):

.. code-block:: bash

   # Create environment from environment.yml
   conda env create -f environment.yml

This process may take 5-15 minutes as it downloads and installs all dependencies (~2GB).

**Step 3: Activate the Environment**

.. code-block:: bash

   # Activate the environment
   conda activate geoworkflow

   # Your prompt should now show (geoworkflow) prefix
   # Example: (geoworkflow) user@computer:~/geoworkflow$

**Step 4: Install GeoWorkflow Package**

.. code-block:: bash

   # Install in development/editable mode
   pip install -e ".[dev]"

**Step 5: Verify Installation**

.. _verify-installation:

.. code-block:: bash

   # Test the installation
   python -c "import geoworkflow; print('GeoWorkflow installed successfully!')"
   
   # Check that key dependencies are available
   python -c "import geopandas, rasterio, xarray; print('Geospatial libraries OK!')"

.. note::

   **First-time Conda users**: The conda environment must be activated every time you open a new terminal session:
   
   .. code-block:: bash

      conda activate geoworkflow

   To check which environment is active: ``conda env list`` (active environment has a * next to it)

Alternative: Using pip Only (Advanced Users)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   Installing with pip alone is **not recommended** for most users, as it requires you to manually install GDAL, GEOS, and PROJ system libraries. Use this only if you cannot use conda.

**For Ubuntu/Debian Linux**:

.. code-block:: bash

   # Install system dependencies
   sudo apt-get update
   sudo apt-get install -y \
       python3-dev \
       libgdal-dev \
       libgeos-dev \
       libproj-dev \
       libspatialindex-dev

   # Clone and install
   git clone https://github.com/jacksonfloods/geoworkflow.git
   cd geoworkflow
   
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install with all dependencies
   pip install -e ".[all]"

**For macOS with Homebrew**:

.. code-block:: bash

   # Install system dependencies
   brew install gdal geos proj

   # Clone and install
   git clone https://github.com/jacksonfloods/geoworkflow.git
   cd geoworkflow
   
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install with all dependencies
   pip install -e ".[all]"

**Windows pip installation is not recommended** - please use conda on Windows.

Troubleshooting Installation
-----------------------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: ``conda: command not found`` after installation

**Solution**:

1. Close and reopen your terminal
2. If still not found, manually add conda to PATH:

   - **Windows**: Add ``C:\Users\YourName\miniconda3\Scripts`` to system PATH
   - **macOS/Linux**: Add ``export PATH="$HOME/miniconda3/bin:$PATH"`` to ``~/.bashrc`` or ``~/.zshrc``

**Issue**: ``Solving environment: failed with initial frozen solve`` 

**Solution**: Try creating the environment with a more relaxed solver:

.. code-block:: bash

   conda config --set channel_priority flexible
   conda env create -f environment.yml

**Issue**: ``ImportError: No module named 'osgeo'`` (GDAL not found)

**Solution**: Ensure you're using the conda environment, not system Python:

.. code-block:: bash

   conda activate geoworkflow
   which python  # Should point to conda environment

**Issue**: Environment creation is very slow or hangs

**Solution**: 

1. Update conda: ``conda update -n base conda``
2. Use mamba (faster solver): 

   .. code-block:: bash

      conda install -n base mamba
      mamba env create -f environment.yml

**Issue**: ``Permission denied`` errors during installation

**Solution**:

- Don't use ``sudo`` with conda
- Ensure you have write permissions to the installation directory
- On Windows, run terminal as Administrator if needed

**Issue**: Conflicts with existing Python installation

**Solution**: Use conda's isolated environment - it won't interfere with system Python:

.. code-block:: bash

   # Check Python location before activation
   which python  # System Python
   
   # Activate conda environment  
   conda activate geoworkflow
   which python  # Should now be conda's Python

Getting Help
~~~~~~~~~~~~

If you encounter issues not covered here:

1. Check existing `GitHub Issues <https://github.com/jacksonfloods/geoworkflow/issues>`_
2. Review the :doc:`../user_guide/how-to/troubleshooting` guide
3. Open a new issue with:

   - Your operating system and version
   - Python version (``python --version``)
   - Conda version (``conda --version``)
   - Complete error message
   - Steps to reproduce

Next Steps
----------

Once installation is complete, proceed to:

* :doc:`quickstart` - Your first GeoWorkflow pipeline
* :doc:`configuration` - Understanding configuration files
* :doc:`first-workflow` - Complete tutorial walkthrough
