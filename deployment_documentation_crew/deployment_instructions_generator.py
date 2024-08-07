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

# Write a file.
def write_utf8_file(output_filepath, content):
    # Get the current datetime with milliseconds
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Split the filepath into directory and filename
    directory, filename = os.path.split(output_filepath)
    
    # Split the filename into name and extension
    name, ext = os.path.splitext(filename)
    
    # Create the new filename with the datetime
    new_filename = f"{name}_{current_time}{ext}"
    
    # Join the directory and new filename
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

    def _generate_description(self) -> None:
        self.description = (
            f"Reads the content of a file from the repository at {self.repo_path}. "
            "Provide the file path relative to the repository root."
        )

class CustomFixedDirectoryReadToolSchema(BaseModel):
    """Input for DirectoryReadTool."""
    ignore_dirs: Optional[List[str]] = Field(default=None, description="List of subdirectories to ignore")

class CustomDirectoryReadToolSchema(CustomFixedDirectoryReadToolSchema):
    """Input for DirectoryReadTool."""
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

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        directory = kwargs.get('directory', self.directory)
        ignore_dirs = kwargs.get('ignore_dirs', self.ignore_dirs)
        
        if directory[-1] == "/":
            directory = directory[:-1]
        
        files_list = []
        for root, dirs, files in os.walk(directory):
            # Remove ignored directories
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

# Crew Definition
class DeploymentInstructionsCrew:
    def __init__(self, repo_path, llm):
        self.repo_path = repo_path
        self.ignore_dirs = ['.git', '.idea', '.vscode', '__pycache__', 'node_modules', 'venv', 'env']
        self.directory_tool = CustomDirectoryReadTool(directory=repo_path, ignore_dirs=self.ignore_dirs)
        self.file_tool = CustomFileReadTool(repo_path=repo_path)
        
        self.llm = llm

        # Initialize agent variables
        self.rpeository_analyzer = None
        self.deployment_specialist = None
        self.technical_writer = None
        self.markdown_writer = None
        
        # Initialize task variables
        self.analyze_repo = None
        self.identify_requirements = None
        self.write_instructions = None
        self.format_and_save = None

        self.create_agents()
        self.create_tasks()

    def create_agents(self):
        file_tool_instruction = (
            "When using the Read File tool, always provide file paths relative to the repository root. "
            "For example, use 'config/webpack.dev.js' instead of '/config/webpack.dev.js' or './config/webpack.dev.js'."
        )
                
        self.repository_analyzer = Agent(
            role='Repository Analyzer',
            goal='Analyze the structure and contents of the code repository, excluding common IDE and version control directories',
            backstory=f'You are an expert in software architecture and code analysis. You can quickly understand the structure of a repository and identify key components, while ignoring non-essential directories such as {", ".join(self.ignore_dirs)}. {file_tool_instruction}',
            tools=[self.directory_tool, self.file_tool],
            verbose=False,
            llm=self.llm
        )

        self.deployment_specialist = Agent(
            role='Deployment Specialist',
            goal='Identify deployment requirements and best practices',
            backstory=f'You are a seasoned DevOps engineer with extensive experience in deploying various types of applications. You understand different deployment strategies and can recommend the best approach based on the application structure. {file_tool_instruction}',
            tools=[self.file_tool],
            verbose=False,
            llm=self.llm
        )

        self.technical_writer = Agent(
            role='Technical Writer',
            goal='Create clear and concise deployment instructions',
            backstory='You are a skilled technical writer with a background in software development. You can translate complex technical information into easy-to-follow instructions for developers of all skill levels.',
            verbose=False,
            llm=self.llm
        )

        self.markdown_writer = Agent(
            role='Markdown Writer',
            goal='Format deployment instructions in markdown format',
            backstory='You are an expert in formatting well-structured markdown documents. You can take technical content and format it into clear, organized markdown.',
            verbose=True,
            llm=self.llm
        )

    def create_tasks(self):
        self.analyze_repo = Task(
            description=f'Analyze the repository structure at {self.repo_path}. Identify the main components, dependencies, and any configuration files related to deployment. The following directories are being ignored: {", ".join(self.ignore_dirs)}.',
            agent=self.repository_analyzer,
            expected_output='A detailed report on the repository structure, highlighting key components and files relevant to deployment, excluding non-essential directories.',
            callback=self.task_callback
        )

        self.identify_requirements = Task(
            description='Based on the repository analysis, identify the deployment requirements. Consider the type of application, its dependencies, and any specific infrastructure needs.',
            agent=self.deployment_specialist,
            expected_output='A list of deployment requirements and recommendations for the best deployment strategy.',
            context=[self.analyze_repo],
            callback=self.task_callback
        )

        self.write_instructions = Task(
            description='Create a comprehensive set of deployment instructions based on the repository analysis and identified requirements. Include steps for setting up the environment, lists of configuration settings, deploying the application, and any post-deployment tasks. Organize your output into logical sections, each starting with a level 2 heading (##).',
            agent=self.technical_writer,
            expected_output='A verbose and detailed, step-by-step guide for deploying the application, including important configuration details and environment variables, formatted in Markdown with clear section headings.',
            context=[self.analyze_repo, self.identify_requirements],
            callback=self.task_callback
        )

        self.format_and_save = Task(
            description='Take the deployment instructions and format them in proper markdown format.',
            agent=self.markdown_writer,
            expected_output='Confirmation that the markdown file has been successfully written with proper formatting and encoding.',
            context=[self.write_instructions],
            callback=self.task_callback
        )

    def task_callback(self, output):
        pass

    def get_all_agents(self):
        """Return a list of all agents in the crew."""
        return [
            self.repository_analyzer,
            self.deployment_specialist,
            self.technical_writer,
            self.markdown_writer
        ]

    def get_all_tasks(self):
        """Return a list of all tasks in the crew."""
        return [
            self.analyze_repo,
            self.identify_requirements,
            self.write_instructions,
            self.format_and_save
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
    output_file = os.getenv('OUTPUT_FILE', "/path/to/output/directory")
    model = os.getenv('LLM_MODEL')
    api_key = os.getenv('LLM_API_KEY')
    temperature = float(os.getenv('LLM_TEMPERATURE', 0))
    llm = get_llm(model, api_key, temperature)
    deployment_crew = DeploymentInstructionsCrew(repo_path, llm)
    result = deployment_crew.run()
    if result.tasks_output and len(result.tasks_output) > 0:
        last_task_output = result.tasks_output[-1]
        write_utf8_file(output_file, last_task_output.raw)
    else:
        write_utf8_file(output_file, result.raw)