import os
import re
from typing import Optional, Type, Any, List
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool, FileReadTool
from langchain_anthropic import ChatAnthropic
from pydantic.v1 import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

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

class MarkdownWriterToolSchema(BaseModel):
    content: str = Field(..., description="The content to be written to markdown files")
    output_dir: str = Field(..., description="The directory where markdown files will be written")


class MarkdownWriterTool(BaseTool):
    name: str = "Markdown Writer"
    description: str = "Writes content to markdown files with proper UTF-8 encoding"
    args_schema: type[BaseModel] = MarkdownWriterToolSchema

    def _run(self, content: str, output_dir: str) -> str:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Split content into sections based on ## headings
        sections = re.split(r'\n##\s', content)

        # Write main file
        main_content = sections[0]
        with open(os.path.join(output_dir, 'deployment_instructions.md'), 'w', encoding='utf-8') as f:
            f.write(main_content)

        # Write individual section files
        for i, section in enumerate(sections[1:], 1):
            # Extract section title from the first line
            section_title = section.split('\n')[0].strip()
            file_name = f"{i:02d}_{section_title.lower().replace(' ', '_')}.md"
            with open(os.path.join(output_dir, file_name), 'w', encoding='utf-8') as f:
                f.write(f"## {section}")

        return f"Deployment instructions have been written to {output_dir}"

class DeploymentInstructionsCrew:
    def __init__(self, repo_path, output_dir):
        self.repo_path = repo_path
        self.ignore_dirs = ['.git', '.idea', '.vscode', '__pycache__', 'node_modules', 'venv', 'env']
        self.directory_tool = CustomDirectoryReadTool(directory=repo_path, ignore_dirs=self.ignore_dirs)
        self.file_tool = FileReadTool()
        self.markdown_writer_tool = MarkdownWriterTool()
        self.output_dir = output_dir
        
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        self.llm = ChatAnthropic(anthropic_api_key=api_key, model="claude-3-sonnet-20240229", temperature=0)

    def create_agents(self):
        repository_analyzer = Agent(
            role='Repository Analyzer',
            goal='Analyze the structure and contents of the code repository, excluding common IDE and version control directories',
            backstory=f'You are an expert in software architecture and code analysis. You can quickly understand the structure of a repository and identify key components, while ignoring non-essential directories such as {", ".join(self.ignore_dirs)}.',
            tools=[self.directory_tool, self.file_tool],
            verbose=True,
            llm=self.llm
        )

        deployment_specialist = Agent(
            role='Deployment Specialist',
            goal='Identify deployment requirements and best practices',
            backstory='You are a seasoned DevOps engineer with extensive experience in deploying various types of applications. You understand different deployment strategies and can recommend the best approach based on the application structure.',
            tools=[self.file_tool],
            verbose=True,
            llm=self.llm
        )

        technical_writer = Agent(
            role='Technical Writer',
            goal='Create clear and concise deployment instructions',
            backstory='You are a skilled technical writer with a background in software development. You can translate complex technical information into easy-to-follow instructions for developers of all skill levels.',
            verbose=True,
            llm=self.llm
        )

        markdown_writer = Agent(
            role='Markdown Writer',
            goal='Format and write deployment instructions in markdown files',
            backstory='You are an expert in creating well-structured markdown documents. You can take technical content and format it into clear, organized markdown files.',
            tools=[self.markdown_writer_tool],
            verbose=True,
            llm=self.llm
        )

        return [repository_analyzer, deployment_specialist, technical_writer, markdown_writer]

    def create_tasks(self):
        analyze_repo = Task(
            description=f'Analyze the repository structure at {self.repo_path}. Identify the main components, dependencies, and any configuration files related to deployment. The following directories are being ignored: {", ".join(self.ignore_dirs)}.',
            agent=self.create_agents()[0],
            expected_output='A detailed report on the repository structure, highlighting key components and files relevant to deployment, excluding non-essential directories.',
            callback=self.task_callback
        )

        identify_requirements = Task(
            description='Based on the repository analysis, identify the deployment requirements. Consider the type of application, its dependencies, and any specific infrastructure needs.',
            agent=self.create_agents()[1],
            expected_output='A list of deployment requirements and recommendations for the best deployment strategy.',
            context=[analyze_repo],
            callback=self.task_callback
        )

        write_instructions = Task(
            description='Create a comprehensive set of deployment instructions based on the repository analysis and identified requirements. Include steps for setting up the environment, deploying the application, and any post-deployment tasks. Organize your output into logical sections, each starting with a level 2 heading (##).',
            agent=self.create_agents()[2],
            expected_output='A detailed, step-by-step guide for deploying the application, formatted in Markdown with clear section headings.',
            context=[analyze_repo, identify_requirements],
            callback=self.task_callback
        )

        format_and_save = Task(
            description=f'Take the deployment instructions, format them properly, and save them as markdown files in {self.output_dir}. Ensure proper UTF-8 encoding and clear organization of content.',
            agent=self.create_agents()[3],
            expected_output='Confirmation that the markdown files have been successfully written with proper formatting and encoding.',
            context=[write_instructions],
            callback=self.task_callback
        )

        return [analyze_repo, identify_requirements, write_instructions, format_and_save]

    def task_callback(self, output):
        print(f"\n{'=' * 40}")
        print(f"Task Completed by: {output.agent}")
        print(f"Task Output:\n{output.raw}")
        print(f"{'=' * 40}\n")

    def run(self):
        crew = Crew(
            agents=self.create_agents(),
            tasks=self.create_tasks(),
            process=Process.sequential,
            verbose=2
        )

        result = crew.kickoff()
        return result

# Usage
if __name__ == "__main__":
    repo_path = os.getenv('REPO_PATH', "/path/to/your/repository")
    output_dir = os.getenv('OUTPUT_DIR', "/path/to/output/directory")
    deployment_crew = DeploymentInstructionsCrew(repo_path, output_dir)
    result = deployment_crew.run()
    print(result)