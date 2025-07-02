from langchain_community.tools  import DuckDuckGoSearchRun
from langchain_community.tools import ShellTool
from dotenv import load_dotenv

load_dotenv()

## Search Tool 
search_tool = DuckDuckGoSearchRun()
results = search_tool.invoke(" Who won the PSL X")
# print(results)

#Shell Tool
shell_tool = ShellTool()
results = shell_tool.invoke("cd C:\\Users\\tahir\\Desktop\\2025 Funavry Internship\\LangChain")
print(results)
results = shell_tool.invoke("whoami")
print(results)

