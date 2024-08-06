import os
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional, Type, Any, List
from crewai_tools.base_tool import BaseTool
from crewai_tools.decorators import tool
from pydantic.v1 import BaseModel, Field


class CustomFileReadToolSchema(BaseModel):
    file_path: str = Field(..., description="The path to the file to read, relative to the repository root")

class CustomFileReadTool(BaseTool):
    name: str = "Read File"
    description: str = "Reads the content of a file from the repository. Provide the file path relative to the repository root."
    args_schema: type[BaseModel] = CustomFileReadToolSchema
    repo_path: str = Field(..., description="The absolute path to the repository root")

    def __init__(self, repo_path: str, **data):
        super().__init__(repo_path=os.path.abspath(repo_path), **data)

    def _run(self, file_path: str) -> str:
        full_path = os.path.join(self.repo_path, file_path.lstrip('/'))
        try:
            with open(full_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return f"Content of {file_path}:\n\n{content}"
        except FileNotFoundError:
            return f"Error: File '{file_path}' not found in the repository."
        except Exception as e:
            return f"Error reading file '{file_path}': {str(e)}"

class CustomFixedDirectoryReadToolSchema(BaseModel):
    ignore_dirs: Optional[List[str]] = Field(default=None, description="List of subdirectories to ignore")

class CustomDirectoryReadToolSchema(CustomFixedDirectoryReadToolSchema):
    directory: str = Field(..., description="Mandatory directory to list content")

class CustomDirectoryReadTool(BaseTool):
    name: str = "List files in directory"
    description: str = "A tool that can be used to recursively list a directory's content."
    args_schema: Type[BaseModel] = CustomDirectoryReadToolSchema
    directory: Optional[str] = None
    ignore_dirs: Optional[List[str]] = None

    def __init__(self, directory: Optional[str] = None, ignore_dirs: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.ignore_dirs = ignore_dirs or []
        if directory is not None:
            self.directory = directory
            self.description = f"A tool that can be used to list {directory}'s content."
            self.args_schema = CustomFixedDirectoryReadToolSchema
            self._generate_description()

    def _run(self, **kwargs: Any) -> Any:
        directory = kwargs.get('directory', self.directory)
        ignore_dirs = kwargs.get('ignore_dirs', self.ignore_dirs)
        
        if directory[-1] == "/":
            directory = directory[:-1]
        
        files_list = []
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for filename in files:
                file_path = os.path.join(root, filename).replace(directory, '').lstrip(os.path.sep)
                files_list.append(f"{directory}/{file_path}")
        
        files = "\n- ".join(files_list)
        return f"File paths: \n- {files}"

    def _generate_description(self) -> None:
        ignore_dirs_str = ", ".join(self.ignore_dirs) if self.ignore_dirs else "None"
        self.description = (
            f"A tool that can be used to list {self.directory}'s content. "
            f"Ignoring subdirectories: {ignore_dirs_str}"
        )

class MarkdownWriterTool(BaseTool):
    name: str = "Markdown Writer"
    description: str = "Write markdown content to files with custom or timestamped filenames in a specified output directory."

    def _run(self, content: str, output_dir: str = "./output", filename: str = None, prefix: str = "") -> str:
        """
        Write the markdown content to a file in the specified output directory.

        Args:
            content (str): The markdown content to be written.
            output_dir (str): The directory where the markdown file will be written.
            filename (str, optional): The desired filename. If not provided, a timestamped filename will be generated.
            prefix (str, optional): A prefix to add to the generated filename if no specific filename is provided.

        Returns:
            str: A message indicating the success of the operation.

        Raises:
            ValueError: If the content is empty, the output directory is invalid, or the filename is invalid.
            OSError: If there's an error creating the directory or writing the file.
        """
        try:
            self._validate_inputs(content, output_dir, filename)
            output_path = Path(output_dir)
            self._ensure_directory(output_path)
            
            if filename:
                file_path = output_path / self._ensure_md_extension(filename)
            else:
                file_path = output_path / self._generate_filename(prefix)

            self._write_content(file_path, content)

            return f"Markdown content has been written to {file_path}"
        except (ValueError, OSError) as e:
            logging.error(f"Error in MarkdownWriterTool: {str(e)}")
            return f"Error writing markdown: {str(e)}"

    def _validate_inputs(self, content: str, output_dir: str, filename: str = None) -> None:
        if not content.strip():
            raise ValueError("Content cannot be empty.")
        if not output_dir:
            raise ValueError("Output directory must be specified.")
        if filename and not filename.strip():
            raise ValueError("Filename cannot be empty if provided.")

    def _ensure_directory(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def _generate_filename(self, prefix: str = "") -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{prefix}{'_' if prefix else ''}{timestamp}.md"

    def _ensure_md_extension(self, filename: str) -> str:
        if not filename.lower().endswith('.md'):
            return f"{filename}.md"
        return filename

    def _write_content(self, file_path: Path, content: str) -> None:
        with file_path.open('w', encoding='utf-8') as f:
            f.write(content)

    def _args_schema(self) -> dict:
        return {
            "content": {
                "type": "string",
                "description": "The markdown content to be written."
            },
            "output_dir": {
                "type": "string",
                "description": "The directory where the markdown files will be written.",
                "default": "./output"
            },
            "filename": {
                "type": "string",
                "description": "The desired filename. If not provided, a timestamped filename will be generated.",
                "default": None
            },
            "prefix": {
                "type": "string",
                "description": "A prefix to add to the generated filename if no specific filename is provided.",
                "default": ""
            }
        }