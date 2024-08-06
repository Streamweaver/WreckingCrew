import os
from datetime import datetime
from typing import Optional, Type, Any, List
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool, tool
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

# LLM Factory
def get_llm(model: str, api_key: str, temperature: float = 0):
    if model.startswith("claude"):
        return ChatAnthropic(anthropic_api_key=api_key, model=model, temperature=temperature)
    elif model.startswith("gpt"):
        return ChatOpenAI(model=model, temperature=temperature, openai_api_key=api_key)
    else:
        raise ValueError(f"Model {model} not supported.")

# Write a file
def write_utf8_file(output_filepath, content):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory, filename = os.path.split(output_filepath)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_{current_time}{ext}"
    new_filepath = os.path.join(directory, new_filename)
    
    try:
        with open(new_filepath, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"File successfully written to {new_filepath}")
    except IOError as e:
        print(f"An error occurred while writing the file: {e}")

# Custom Tools

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

# Codebase Documentation Crew
class CodebaseDocumentationCrew:
    def __init__(self, repo_path, llm):
        self.repo_path = repo_path
        self.ignore_dirs = ['.git', '.idea', '.vscode', '__pycache__', 'node_modules', 'venv', 'env']
        self.directory_tool = CustomDirectoryReadTool(directory=repo_path, ignore_dirs=self.ignore_dirs)
        self.file_tool = CustomFileReadTool(repo_path=repo_path)
        
        self.llm = llm

        # Initialize agent variables
        self.repository_analyzer = None
        self.code_reviewer = None
        self.documentation_writer = None
        self.markdown_formatter = None
        
        # Initialize task variables
        self.analyze_repo_structure = None
        self.review_code_components = None
        self.write_documentation = None
        self.format_documentation = None

        self.create_agents()
        self.create_tasks()

    def create_agents(self):
        file_tool_instruction = (
            "When using the Read File tool, always provide file paths relative to the repository root. "
            "For example, use 'src/main.py' instead of '/src/main.py' or './src/main.py'."
        )
                
        self.repository_analyzer = Agent(
            role='Repository Analyzer',
            goal='Analyze the structure and organization of the codebase',
            backstory=f'You are an expert in software architecture and code organization. You can quickly understand the structure of a repository and identify key components, while ignoring non-essential directories such as {", ".join(self.ignore_dirs)}. {file_tool_instruction}',
            tools=[self.directory_tool, self.file_tool],
            verbose=False,
            llm=self.llm
        )

        self.code_reviewer = Agent(
            role='Code Reviewer',
            goal='Analyze code components and identify main features and functionalities',
            backstory=f'You are a senior software engineer with extensive experience in code review and analysis. You can quickly understand complex codebases and identify key features, design patterns, and architectural decisions. {file_tool_instruction}',
            tools=[self.file_tool],
            verbose=False,
            llm=self.llm
        )

        self.documentation_writer = Agent(
            role='Documentation Writer',
            goal='Create comprehensive and clear documentation for the codebase',
            backstory='You are a skilled technical writer with a strong background in software development. You can translate complex technical information into clear, concise, and well-structured documentation that is easily understood by developers of all skill levels.',
            verbose=False,
            llm=self.llm
        )

        self.markdown_formatter = Agent(
            role='Markdown Formatter',
            goal='Format the codebase documentation in proper markdown format',
            backstory='You are an expert in creating well-structured and visually appealing markdown documents. You can take technical content and format it into clear, organized, and easily navigable markdown documentation.',
            verbose=True,
            llm=self.llm
        )

    def create_tasks(self):
        self.analyze_repo_structure = Task(
            description=f'Analyze the repository structure at {self.repo_path}. Identify the main directories, key files, and overall organization of the codebase. The following directories are being ignored: {", ".join(self.ignore_dirs)}.',
            agent=self.repository_analyzer,
            expected_output='A detailed report on the repository structure, highlighting the main components, important files, and overall architecture of the codebase.',
            callback=self.task_callback
        )

        self.review_code_components = Task(
            description='Based on the repository analysis, review the main code components. Identify key features, important classes and functions, design patterns used, and any notable architectural decisions.',
            agent=self.code_reviewer,
            expected_output='A comprehensive analysis of the codebase, detailing main features, detailed descriptions of each important component, and architectural insights.',
            context=[self.analyze_repo_structure],
            callback=self.task_callback
        )

        self.write_documentation = Task(
            description='Create comprehensive documentation for the codebase based on the repository analysis and code review. Include an overview of the project structure, detailed explanations of main features and components, usage instructions, and any important technical decisions or patterns used. Organize your output into logical sections, each starting with a level 2 heading (##).',
            agent=self.documentation_writer,
            expected_output='A detailed and well-structured documentation of the codebase, including project overview, main features, code organization, and usage instructions, formatted with clear section headings. Also include detailed descirptions of each individual component.',
            context=[self.analyze_repo_structure, self.review_code_components],
            callback=self.task_callback
        )

        self.format_documentation = Task(
            description='Take the codebase documentation and format it in proper markdown format. Ensure the document is well-structured, easily navigable, and visually appealing.',
            agent=self.markdown_formatter,
            expected_output='A beautifully formatted markdown file containing the comprehensive codebase documentation.',
            context=[self.write_documentation],
            callback=self.task_callback
        )

    def task_callback(self, output):
        pass

    def get_all_agents(self):
        return [
            self.repository_analyzer,
            self.code_reviewer,
            self.documentation_writer,
            self.markdown_formatter
        ]

    def get_all_tasks(self):
        return [
            self.analyze_repo_structure,
            self.review_code_components,
            self.write_documentation,
            self.format_documentation
        ]

    def run(self):
        crew = Crew(
            agents=self.get_all_agents(),
            tasks=self.get_all_tasks(),
            process=Process.sequential,
            verbose=1
        )

        crew_result = crew.kickoff()
        return crew_result

# Usage
if __name__ == "__main__":
    repo_path = os.getenv('REPO_PATH', "/path/to/your/repository")
    output_file = os.getenv('OUTPUT_FILE', "/path/to/output/directory/codebase_documentation.md")
    model = os.getenv('LLM_MODEL')
    api_key = os.getenv('LLM_API_KEY')
    temperature = float(os.getenv('LLM_TEMPERATURE', 0))
    llm = get_llm(model, api_key, temperature)
    documentation_crew = CodebaseDocumentationCrew(repo_path, llm)
    result = documentation_crew.run()

    # Robust error handling for crew output
    if hasattr(result, 'tasks_output') and result.tasks_output:
        # If tasks_output exists and is non-empty, use the last task's output
        last_task_output = result.tasks_output[-1]
        if hasattr(last_task_output, 'raw'):
            write_utf8_file(output_file, last_task_output.raw)
        else:
            write_utf8_file(output_file, str(last_task_output))
    elif hasattr(result, 'raw'):
        # If there's no tasks_output but there is a raw attribute, use that
        write_utf8_file(output_file, result.raw)
    else:
        # If neither of the above work, convert the entire result to a string
        write_utf8_file(output_file, str(result))

    print(f"Result type: {type(result)}")
    print(f"Result attributes: {dir(result)}")