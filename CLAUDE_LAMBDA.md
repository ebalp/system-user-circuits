### New instance bootstrap

When a user asks to "set up the instance" or "clone the repo and set up", follow these steps:

1. **Clone the repo to local disk:**
   ```bash
   cd /home/ubuntu
   git clone https://github.com/ebalp/system-user-circuits.git
   cd system-user-circuits
   ```

2. **Locate the config file:**
   ```bash
   ls *.sync.env
   ```
   If none exists, ask the user: "Do you have a `.sync.env` config file? You can upload it through Jupyter to the repo directory. Otherwise I can create one from the template if you give me the values."

   - **If they can upload it**: Wait for them to upload it into the repo directory, then continue.
   - **If they need to create it**: Ask for the values listed in `sync.env.template` and create `config.sync.env` from them.

3. **Run setup** to configure git and install the Python environment (uv, Python 3.12, and `uv sync` are all handled automatically). Explain what will happen to the user:
   ```bash
   ./lambda-sync.sh <config>.sync.env setup
   ```

4. **Download from bucket** (if the user has previous work): follow the sync protocol below.

### Running sync commands (upload / download)

The sync script has one confirmation prompt for both upload and download. Claude Code cannot handle interactive prompts natively, so the protocol is:

1. **Explain what will happen** before running anything. Be specific:
   - Which paths will be synced (explicit paths given, or auto-discovery of all */data/ and */reports/)
   - That patterns in `.syncignore` won't be synced 
   - For **upload**: bucket data will be overwritten with local files
   - For **download**: local files will be overwritten with bucket data — unsynced local changes will be lost

2. **Wait for the user to confirm** in the conversation.

3. **Only after confirmation**, pipe the response:
   ```bash
   # Full sync
   printf "y\n" | bash ./lambda-sync.sh <config>.sync.env upload
   printf "y\n" | bash ./lambda-sync.sh <config>.sync.env download

   # Targeted sync (one or more paths)
   printf "y\n" | bash ./lambda-sync.sh <config>.sync.env upload phase0/data/results
   printf "y\n" | bash ./lambda-sync.sh <config>.sync.env download phase0/data
   ```

### Before shutting down

Always remind the user to upload before terminating. Follow the sync protocol above.