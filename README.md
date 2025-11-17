# pydantic-store

A collection of Pydantic models and storage approaches for ease of reading to/from files or creating caches. 

## Installation

```bash
pip install pydantic-store
```

### Optional Dependencies

Additional dependencies needed for storing as YAML or TOML (not included to keep the dependency limited to pydantic)

```bash
# For YAML support
pip install ruamel.yaml

# For TOML writing support  
pip install tomli-w

# For TOML reading on Python < 3.11
pip install tomli
```

## Quick Start

```python
from pydantic_store import BaseModel, ListModel, DictModel, JsonStore, PydanticDBM

# Define your models
class User(BaseModel):
    name: str
    age: int
    email: str

# Create and save to file
user = User(name="Alice", age=30, email="alice@example.com")
user.to_file("user.json")

# Load from file
loaded_user = User.from_file("user.json")
```

## Model Types

### BaseModel

Enhanced Pydantic BaseModel with file I/O capabilities.

```python
from pydantic_store import BaseModel

class Config(BaseModel):
    database_url: str
    debug: bool = False
    max_connections: int = 10

# Save to different formats
config = Config(database_url="postgresql://localhost/mydb")
config.to_file("config.json")        # JSON format
config.to_file("config.yaml")        # YAML format  
config.to_file("config.toml")        # TOML format

# Load from file (format auto-detected by extension)
config = Config.from_file("config.yaml")
```

### RootModel

A generic root model for wrapping single values with validation.

```python
from pydantic_store import RootModel

class Port(RootModel[int]):
    pass

port = Port(8080)
port.to_file("port.json")  # Saves: 8080
loaded_port = Port.from_file("port.json")
```

### ListModel

A list-like model that behaves like a Python list while providing Pydantic validation.

```python
from pydantic_store import ListModel

TodoList = ListModel[str]

todos = TodoList(["Buy groceries", "Walk the dog"])

# Use like a regular list
todos.append("Read a book")
todos.extend(["Exercise", "Cook dinner"])
print(len(todos))  # 5
print(todos[0])    # "Buy groceries"

# Persist to file
todos.to_file("todos.json")

# Load from file
loaded_todos = TodoList.from_file("todos.json")
```

### DictModel  

A dictionary-like model that behaves like a Python dict with validation.

```python
from pydantic_store import DictModel

Settings = DictModel[str, int]

settings = Settings({"timeout": 30, "retries": 3})

# Use like a regular dict
settings["max_workers"] = 4
settings.update({"cache_size": 1000})
print(settings.keys())
print(len(settings))

# Persist to file
settings.to_file("settings.yaml")
```

### PydanticDBM

A wrapper around the sqlite DBM backend (backported from 3.14) to store and retrieve pydantic models. 

```python
from pydantic_store import PydanticDBM
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

UserDBM = PydanticDBM[User]

# Method 1: Type subscription
with UserDBM("users.db") as db:
    user = User(name="Alice", age=30)
    db["alice"] = user
    retrieved_user = db["alice"]  # Automatically validated as User

# Method 2: Explicit storage format
with PydanticDBM("users.db", storage_format=User) as db:
    db["bob"] = User(name="Bob", age=25)
```

### JsonStore

A persistent dictionary that automatically saves changes to disk.

```python
from pydantic_store import JsonStore
from pathlib import Path

# Connect to a JSON file (creates if doesn't exist)
store = JsonStore[str].connect(Path("data.json"))

# Changes are automatically persisted
store["user:1"] = "Alice"
store["user:2"] = "Bob" 

# Data is immediately written to data.json
print(store["user:1"])  # "Alice"
```

### Supported Formats

pydantic-store supports multiple file formats with automatic format detection:

- **JSON**: Always available
- **YAML**: Human-readable format (requires `ruamel.yaml`)
- **TOML**: Configuration-friendly format (requires `tomli` for reading, `tomli-w` for writing)


You can also specify the format:

```python
model.to_file("data.txt", file_format="json")
```

## Licence

MIT Licence - see LICENSE.md for details.