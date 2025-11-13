import shutil
import os
from pathlib import Path

# transpose github workflows into cookiecutter after generate
# there's a problem where a github actions self template can't 
# modify an action, so we instead have all actions at the top - so
# they are only removed, and not 'modified' by the templating process.
# this hook makes sure when used as a cookiecutter - the actions are loaded correctly.

template_dir = r"{{ cookiecutter._template }}"
if any(x in template_dir for x in ["https:", "gh:"]):
    repo_name = template_dir.split("/")[-1]
    template_dir = Path.home() / ".cookiecutters" / repo_name
else:
    template_dir = Path(template_dir)

workflow_dir_source = template_dir / ".github" / "workflows"
workflow_dir_dest = Path(os.getcwd()) / ".github" / "workflows"

workflow_dir_dest.mkdir(parents=True, exist_ok=True)
for item in workflow_dir_source.glob("*.yml"):
    if item.name.startswith("template_") is False:
        shutil.copyfile(item, workflow_dir_dest / item.name)