<h1 align="center">petprep</h1>

<p align="center">
  PETPrep: A Robust Preprocessing Pipeline for PET Data
</p>

<p align="center">
  <a href="https://github.com/brainlife/abcd-spec"><img src="https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg" alt="ABCD Spec v1.1"></a>
  <a href="https://doi.org/10.25663/brainlife.app.878"><img src="https://img.shields.io/badge/Brainlife-brainlife.app.878-blue.svg" alt="Run on brainlife.io"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-see_LICENSE-lightgrey.svg" alt="License"></a>
</p>

## At a glance

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **PETPrep: A Robust Preprocessing Pipeline for PET Data**

| | |
|---|---|
| **Run it** | [doi.org/10.25663/brainlife.app.878](https://doi.org/10.25663/brainlife.app.878) |

<!-- brainlife-app:end -->

## Running the App

### On brainlife.io

Open the App at [doi.org/10.25663/brainlife.app.878](https://doi.org/10.25663/brainlife.app.878), choose the input datasets from your project and press **Execute**.

### Locally

You need [`jq`](https://jqlang.github.io/jq/).

1. Clone this repository.
2. Create `config.json` with the paths to your inputs and the parameters:

   ```json
   {}
   ```

3. Run `./main`.

## Citing

If you use this App, please cite it, the software it builds on and brainlife.io:

- petprep. brainlife.io App. [doi:10.25663/brainlife.app.878](https://doi.org/10.25663/brainlife.app.878)
- Hayashi S, Caron BA, Heinsfeld AS, et al. brainlife.io: a decentralized and open-source cloud platform to support neuroscience research. *Nature Methods* 21:809–813 (2024). [doi:10.1038/s41592-024-02237-2](https://doi.org/10.1038/s41592-024-02237-2)
- Avesani P, McPherson B, Hayashi S, et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. *Scientific Data* 6:69 (2019). [doi:10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

## Authors

Maintained by [@nipreps](https://github.com/nipreps).

## License

See [LICENSE](LICENSE).
