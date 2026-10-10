Development Container Setup
============================

This directory contains configuration files for running PETPrep in a Docker-based development container.
The devcontainer provides a pre-configured environment with all PETPrep dependencies, allowing you to
develop and test PETPrep without manually setting up the environment. For more information about devcontainers, see the `VS Code Dev Containers documentation
<https://code.visualstudio.com/docs/devcontainers/containers>`__. or `Containers.dev <https://containers.dev/>`__.

Prerequisites
-------------

* VS Code or a VS Code-based IDE (such as Cursor, GitHub Codespaces, etc.)
* Docker Desktop installed and running
* The `Dev Containers` extension installed in VS Code
* A FreeSurfer license file (obtain one for free at https://surfer.nmr.mgh.harvard.edu/registration.html)

Quick Start
-----------

1. **Copy the example configuration:**

   .. code-block:: bash

      cp .devcontainer/example.devcontainer.json .devcontainer/devcontainer.json

2. **Edit the mount points in ``.devcontainer/devcontainer.json``:**

   Open the file and update the following lines in the ``mounts`` array:

   * **Data directory mount** (line 8): Replace ``/FullPath/to/the/data/directory/contaning/your/bids/data``
     with the absolute path to directory containing bids data on your host machine.

   * **FreeSurfer license mount** (line 9): Replace ``/FullPath/to/your/freesurfer/license.txt``
     with the absolute path to your FreeSurfer license file on your host machine.

   Example for macOS:

   .. code-block:: json

      "mounts": [
          "source=/Users/yourusername/Data/my_bids_dataset,target=/data,type=bind,consistency=cached",
          "source=/Applications/freesurfer/7.4.1/license.txt,target=/opt/freesurfer/license.txt,type=bind,consistency=cached,readonly=true"
      ]

   Example for Linux:

   .. code-block:: json

      "mounts": [
          "source=/home/yourusername/data/my_bids_dataset,target=/data,type=bind,consistency=cached",
          "source=/opt/freesurfer/license.txt,target=/opt/freesurfer/license.txt,type=bind,consistency=cached,readonly=true"
      ]

3. **Open the project in VS Code:**

   * Open VS Code
   * Open the PETPrep repository folder
   * VS Code should detect the ``.devcontainer/devcontainer.json`` file and prompt you to reopen in container
   * Click "Reopen in Container" when prompted

   Alternatively, you can manually trigger the container:

   * Press ``Ctrl+Shift+P`` (or ``Cmd+Shift+P`` on macOS) to open the command palette
   * Type "Dev Containers: Reopen in Container" and select it

4. **Wait for the container to build and initialize:**

   * The first time will take longer as Docker downloads the PETPrep image
   * The ``postCreateCommand`` will install PETPrep in editable mode with development dependencies
   * You'll see progress in the VS Code terminal

What's Included
---------------

The devcontainer provides:

* **Pre-configured Python environment** with all PETPrep dependencies
* **Development tools**: ruff (linter), black (formatter), pytest (testing)
* **VS Code extensions**: Python, Pylance, Ruff, Black Formatter
* **Mounted data directory**: Your BIDS dataset accessible at ``/data``
* **FreeSurfer license**: Automatically mounted for FreeSurfer workflows
* **TemplateFlow cache**: Pre-cached templates at ``/home/petprep/.cache/templateflow``

Configuration Details
---------------------

The devcontainer configuration includes:

* **Image**: Uses the official PETPrep Docker image (``ghcr.io/nipreps/petprep:main``)
* **User**: Runs as ``petprep`` user (non-root for security)
* **Workspace**: Your code is mounted at ``/workspace``
* **Python path**: ``/opt/conda/envs/petprep/bin/python``
* **Environment variables**:
  * ``TEMPLATEFLOW_HOME``: Set to ``/home/petprep/.cache/templateflow``
  * ``PYTHONPATH``: Includes ``/workspace`` for editable installs

Troubleshooting
---------------

**Container won't start:**
  * Ensure Docker Desktop is running
  * Check that the paths in your ``devcontainer.json`` are correct and accessible
  * Try rebuilding: Command Palette → "Dev Containers: Rebuild Container"

**Permission errors:**
  * The container runs as the ``petprep`` user
  * If you need to install packages, use ``--user`` flag: ``pip install --user package_name``

**TemplateFlow path errors:**
  * If you see errors about templateflow paths from your host machine, delete the ``work/`` directory:
    ``rm -rf work/``
  * The container has templates pre-cached, so you don't need to mount your host templateflow cache

**FreeSurfer license not found:**
  * Verify the license file path in ``devcontainer.json`` is correct
  * Ensure the file exists and is readable
  * Check that the mount uses ``readonly=true`` for security

**Package installation conflicts:**
  * The container already has all runtime dependencies installed
  * Use ``--no-deps`` when installing packages to avoid conflicts
  * Development dependencies are installed via ``postCreateCommand``

Rebuilding the Container
------------------------

If you need to rebuild the container (e.g., after changing ``devcontainer.json``):

1. Command Palette → "Dev Containers: Rebuild Container"
2. Or for a clean rebuild: "Dev Containers: Rebuild Container Without Cache"

The container will be rebuilt and the ``postCreateCommand`` will run again.

Additional Notes
----------------

* The ``devcontainer.json`` file is gitignored to prevent committing personal paths
* The ``example.devcontainer.json`` serves as a template for new users
* All changes made inside the container persist until you rebuild it
* The container uses the same image as the production PETPrep Docker image for consistency
* If you need to change the PETPrep version, you can change the image in the ``devcontainer.json`` file or 
rebuild and tag the container with the same name and tag as the official PETPrep Docker image (``ghcr.io/nipreps/petprep:main``).

For more information about devcontainers, see the `VS Code Dev Containers documentation
<https://code.visualstudio.com/docs/devcontainers/containers>`__.
