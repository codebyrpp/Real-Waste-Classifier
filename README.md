# Real-Waste-Classifier

This project is for building a CNN classifier to classify real waste.

## Branching Strategy

To ensure smooth collaboration and maintain a stable codebase, we will follow a simple branching strategy.

### Branches

- **`main`**: This branch holds the most stable and production-ready version of the notebook. No direct commits should be made to this branch. Merges to `main` should only come from the `develop` branch after thorough testing.
- **`develop`**: This is the primary development branch where all completed features are integrated. All feature branches are created from `develop` and merged back into it.
- **`feature/<feature-name>`**: When you start working on a new feature or experiment (e.g., `feature/add-new-layer`, `feature/try-new-optimizer`), create a new branch from `develop`. Name it descriptively.

### Quick Workflow (Edit in Colab and commit to your branch)

1.  **Create your feature branch (once):**
    Pull latest `develop` and create a branch.

    ```bash
    git checkout develop
    git pull origin develop
    git checkout -b feature/<your-feature-name>
    git push -u origin feature/<your-feature-name>
    ```

2.  **Open the notebook in Colab on your branch:**

    - In Colab: `File` > `Open notebook` > `GitHub` tab → paste your repo URL → pick `feature/<your-feature-name>` → select `notebook.ipynb`.
    - Or open directly via URL (replace placeholders):

      `https://colab.research.google.com/github/<your-username>/Real-Waste-Classifier/blob/feature/<your-feature-name>/notebook.ipynb`

3.  **Edit as usual in Colab.**

4.  **Commit back to the same branch from Colab:**

    - `File` > `Save a copy in GitHub`
    - Select the same repository and `feature/<your-feature-name>` branch
    - File path: `notebook.ipynb` (overwrite existing)
    - Add a concise commit message → `OK`

5.  **Open a PR when ready:**
    Create a PR from `feature/<your-feature-name>` → `develop`. After review/merge to `develop`, maintainers will promote to `main` for releases.

### Colab tips

- **Branch matters:** Always ensure the `GitHub` tab in Colab shows your `feature/<your-feature-name>` branch before opening the notebook.
- **Auth once:** The first time you save from Colab to GitHub, authorize access; subsequent saves are one-click.
- **Overwrite intentionally:** Keep the path as `notebook.ipynb` on your branch to update the same file each commit.
