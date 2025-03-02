# /// script
# dependencies = [
#   "openai>=1.0.0",
#   "rich",
# ]
# ///

import sys
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown

# Initialize rich console for nice formatting
console = Console()

# Configure OpenAI client to use the local proxy
# The proxy.go default port is 9876
client = OpenAI(
    # api_key="your-replicate-api-key-here",  # Your Replicate API key
    base_url="http://localhost:9876/v1",    # Point to the local proxy
)

def stream_chat_completion():
    """
    Demonstrate streaming chat completion through the proxy
    """
    console.print("[bold green]Starting streaming chat completion demo[/bold green]")
    console.print("[bold yellow]This will connect to the local proxy at localhost:9876[/bold yellow]")
    
    # Create a streaming chat completion
    stream = client.chat.completions.create(
        model="claude-3.7-sonnet",  # This gets mapped by the proxy
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Explain quantum computing in simple terms, one paragraph at a time."}
        ],
        stream=True,  # Enable streaming
        max_tokens=1024
    )
    
    # Process the streaming response
    console.print("\n[bold blue]Response:[/bold blue]")
    
    # Display each chunk as it arrives
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            full_response += content
            console.print(content, end="")
            sys.stdout.flush()  # Make sure output is flushed immediately
    
    console.print("\n\n[bold green]Streaming complete![/bold green]")

def non_streaming_chat_completion():
    """
    Demonstrate non-streaming chat completion through the proxy for comparison
    """
    console.print("\n[bold green]Starting non-streaming chat completion demo[/bold green]")
    
    # Create a non-streaming chat completion
    response = client.chat.completions.create(
        model="claude-3.7-sonnet",  # This gets mapped by the proxy
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Write a short poem about coding."}
        ],
        stream=False,
        max_tokens=200
    )
    
    # Display the response
    console.print("\n[bold blue]Response:[/bold blue]")
    console.print(Markdown(response.choices[0].message.content))
    
    console.print("[bold green]Non-streaming complete![/bold green]")

if __name__ == "__main__":
    # First check if the API key is set
    if client.api_key == "your-replicate-api-key-here":
        console.print(
            "[bold red]Please replace 'your-replicate-api-key-here' with your actual Replicate API key[/bold red]"
        )
        console.print("You can set it in the script or use:")
        console.print("[bold]export OPENAI_API_KEY=your-replicate-api-key[/bold]")
        sys.exit(1)
    
    # Check if proxy is running
    console.print("[bold]Checking if proxy is running on localhost:9876...[/bold]")
    
    try:
        # Run the streaming demo
        stream_chat_completion()
        
        # Run the non-streaming demo
        non_streaming_chat_completion()
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        console.print("\n[yellow]Make sure the proxy server is running with:[/yellow]")
        console.print("[bold]go run proxy.go[/bold]") 
