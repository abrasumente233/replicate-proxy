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

console = Console()
client = OpenAI(base_url="http://localhost:9876/v1")


def stream_chat_completion():
    console.print("[bold green]Starting streaming chat completion demo[/bold green]")
    console.print(
        "[bold yellow]This will connect to the local proxy at localhost:9876[/bold yellow]"
    )

    # Create a streaming chat completion
    stream = client.chat.completions.create(
        model="openai/gpt-4.1",  # This gets mapped by the proxy
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {
                "role": "user",
                "content": "Write Hello World in Rust.",
            },
            {
                "role": "assistant",
                "content": "```rust",
            },
        ],
        stream=True,  # Enable streaming
        max_tokens=1024,
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
    console.print(
        "\n[bold green]Starting non-streaming chat completion demo[/bold green]"
    )

    # Create a non-streaming chat completion
    response = client.chat.completions.create(
        model="openai/gpt-4.1",  # This gets mapped by the proxy
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Write a short poem about coding."},
        ],
        stream=False,
        max_tokens=1024,
    )

    # Display the response
    console.print("\n[bold blue]Response:[/bold blue]")
    console.print(Markdown(response.choices[0].message.content))

    console.print("[bold green]Non-streaming complete![/bold green]")


if __name__ == "__main__":
    # Check if proxy is running
    console.print("[bold]Checking if proxy is running on localhost:9876...[/bold]")

    try:
        stream_chat_completion()
        non_streaming_chat_completion()

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        console.print("\n[yellow]Make sure the proxy server is running with:[/yellow]")
        console.print("[bold]go run proxy.go[/bold]")
