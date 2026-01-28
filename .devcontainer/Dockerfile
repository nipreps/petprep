FROM ghcr.io/nipreps/petprep:main

# Install sudo, git, vim, and htop (add any other dev or quality of life tools you'd like here)
RUN apt-get install -y sudo git vim htop


# Add petprep user to sudoers group and configure passwordless sudo for a more useful dev environment
RUN if id -u petprep > /dev/null 2>&1; then \
        usermod -aG sudo petprep 2>/dev/null || usermod -aG wheel petprep 2>/dev/null || true; \
        echo "petprep ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/petprep; \
        chmod 0440 /etc/sudoers.d/petprep; \
    fi
