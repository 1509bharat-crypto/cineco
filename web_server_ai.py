#!/usr/bin/env python3
"""
AI-powered web server for Internet Archive Explorer
Integrates ChatGPT for natural language interactions
"""
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import requests
import os
from urllib.parse import urlparse, parse_qs
from openai import OpenAI

# Initialize OpenAI client - set OPENAI_API_KEY environment variable
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# YouTube API Key - set YOUTUBE_API_KEY environment variable
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")

class IAHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        """Handle POST requests for AI chat"""
        if self.path == '/api/chat':
            self.handle_chat()
        else:
            self.send_error(404, "Not found")

    def do_GET(self):
        parsed_path = urlparse(self.path)

        # Handle API requests
        if parsed_path.path == '/api/search':
            self.handle_search(parsed_path)
        elif parsed_path.path == '/api/item':
            self.handle_item(parsed_path)
        else:
            # Serve static files
            super().do_GET()

    def handle_chat(self):
        """Handle AI chat requests"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            user_message = data.get('message', '')
            conversation_history = data.get('history', [])

            if not user_message:
                self.send_error(400, "Missing message")
                return

            # Define tools for ChatGPT
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "search_archive",
                        "description": "Search the Internet Archive for books, movies, audio, and other media",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query"
                                }
                            },
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_item_details",
                        "description": "Get detailed information about a specific Internet Archive item including metadata and download links",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "identifier": {
                                    "type": "string",
                                    "description": "The Internet Archive item identifier"
                                }
                            },
                            "required": ["identifier"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "curate_quality_movies",
                        "description": "Get a curated list of high-quality, family-friendly, license-free movies. Excludes adult content and filters by popularity, reviews, and ratings.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "min_views": {
                                    "type": "integer",
                                    "description": "Minimum number of views/downloads (default: 10000)",
                                    "default": 10000
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Maximum number of movies to return (default: 20)",
                                    "default": 20
                                }
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search_youtube",
                        "description": "Search YouTube for videos. Returns video titles, descriptions, channels, and embed URLs.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query for YouTube videos"
                                },
                                "max_results": {
                                    "type": "integer",
                                    "description": "Maximum number of results (default: 10)",
                                    "default": 10
                                }
                            },
                            "required": ["query"]
                        }
                    }
                }
            ]

            # Build messages array with system prompt + conversation history
            system_prompt = """You're Cineco - a friend who really gets film and loves helping people find something perfect to watch.

Talk like a real person. Be natural, warm, curious. Think of this like texting someone you care about, not conducting an interview.

Keep it SHORT. One sentence, maybe two. That's it.

Your whole thing is reading between the lines - understanding the feeling behind what they're saying, not just the words.

How to talk:
- Start where they are. If they mention a mood, reflect it back
- Ask simple questions that dig deeper
- Don't overthink it - just be genuinely curious
- Use "I" and "you" naturally - this is a conversation between two people
- Read the room - if they give short answers, maybe they need options. If they elaborate, ask more

Don't ever:
- Give them a list of anything
- Explain what genres are or define film terms
- Sound like a customer service bot
- Use formal language
- Write paragraphs

After 2-3 exchanges, just check: "Want me to find something?" or "Ready?" Keep it simple.

When you search:
Just say WHY in a human way. One sentence about the feeling, not the films.
Example: "Found some films with that quiet, contemplative vibe you're after."

Examples of how to sound:

Them: "I'm tired"
You: "Need something easy, or something that'll pull you out of it?"

Them: "feeling nostalgic"
You: "What kind? Like childhood stuff, or more recent memories?"

Them: "idk surprise me"
You: "Okay - what's your energy like right now? Up or down?"

Them: "want something cozy"
You: "Cozy and heartwarming, or cozy and contemplative?"

Them: "yes find something"
You: [search] "Got some films that have that warm, settled feeling."

Just be real. No scripts, no formality. Like you're helping a friend figure out what to watch."""

            # Start with system prompt
            messages = [{"role": "system", "content": system_prompt}]

            # Add conversation history (excluding the current message which is already in history)
            # Only use history up to the last message (since current message is already added by frontend)
            if conversation_history:
                messages.extend(conversation_history)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            assistant_message = response.choices[0].message

            # Check if ChatGPT wants to call a function
            if assistant_message.tool_calls:
                # Execute the function call
                tool_call = assistant_message.tool_calls[0]
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                if function_name == "search_archive":
                    function_response = self.search_archive(function_args["query"])
                elif function_name == "get_item_details":
                    function_response = self.get_item_details(function_args["identifier"])
                elif function_name == "curate_quality_movies":
                    min_views = function_args.get("min_views", 10000)
                    limit = function_args.get("limit", 20)
                    function_response = self.curate_quality_movies(min_views, limit)
                elif function_name == "search_youtube":
                    query = function_args["query"]
                    max_results = function_args.get("max_results", 10)
                    function_response = self.search_youtube(query, max_results)
                else:
                    function_response = {"error": "Unknown function"}

                # Send function result back to ChatGPT
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [{
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": tool_call.function.arguments
                        }
                    }]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(function_response)
                })

                # Get final response from ChatGPT
                final_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages
                )

                # Always return the structured data for video/movie searches
                # The frontend will handle displaying it
                result = {
                    "response": final_response.choices[0].message.content,
                    "data": function_response if isinstance(function_response, list) else None
                }
            else:
                # No function call, just return the response
                result = {
                    "response": assistant_message.content,
                    "data": None
                }

            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        except Exception as e:
            print(f"Error in chat: {e}")
            self.send_error(500, f"Server error: {str(e)}")

    def search_archive(self, query):
        """Search Internet Archive with improved filtering"""
        try:
            url = "https://archive.org/advancedsearch.php"

            # Enhanced query with video filtering and adult content exclusion
            enhanced_query = (
                f'({query}) AND '
                'mediatype:movies AND '
                'format:"h.264" AND '
                '-subject:"adult" AND '
                '-subject:"erotica" AND '
                '-subject:"xxx"'
            )

            params = {
                "q": enhanced_query,
                "output": "json",
                "rows": 15,
                "sort": "downloads desc",
                "fl": "identifier,title,description,year,downloads,avg_rating,subject"
            }
            response = requests.get(url, params=params)
            data = response.json()

            docs = data.get("response", {}).get("docs", [])
            results = []

            for d in docs:
                # Additional content filtering
                subjects = d.get("subject", [])
                if isinstance(subjects, str):
                    subjects = [subjects]

                # Skip if contains adult-related keywords
                adult_keywords = ['adult', 'xxx', 'erotica', 'pornography', 'erotic']
                if any(keyword in ' '.join(subjects).lower() for keyword in adult_keywords):
                    continue

                identifier = d.get("identifier")
                results.append({
                    "identifier": identifier,
                    "title": d.get("title"),
                    "description": (d.get("description", "No description available")[:200] + "...") if d.get("description") else "No description available",
                    "year": d.get("year"),
                    "downloads": d.get("downloads", 0),
                    "avg_rating": round(d.get("avg_rating", 0), 1),
                    "thumbnail": f"https://archive.org/services/img/{identifier}",
                    "watch_url": f"https://archive.org/details/{identifier}",
                    "embed_url": f"https://archive.org/embed/{identifier}",
                    "platform": "Internet Archive",
                    "platform_short": "IA"
                })

            return results
        except Exception as e:
            return {"error": str(e)}

    def get_item_details(self, identifier):
        """Get item details from Internet Archive"""
        try:
            url = f"https://archive.org/metadata/{identifier}"
            response = requests.get(url)
            data = response.json()

            meta = data.get("metadata", {})
            files = data.get("files", [])

            return {
                "title": meta.get("title"),
                "description": meta.get("description"),
                "year": meta.get("year"),
                "files": [
                    {
                        "name": f.get("name"),
                        "format": f.get("format"),
                        "url": f"https://archive.org/download/{identifier}/{f.get('name')}"
                    }
                    for f in files[:10]  # Limit to 10 files
                ]
            }
        except Exception as e:
            return {"error": str(e)}

    def curate_quality_movies(self, min_views=5000, limit=20):
        """Curate high-quality, family-friendly movies"""
        try:
            url = "https://archive.org/advancedsearch.php"

            query = (
                'mediatype:movies AND '
                'format:"h.264" AND '
                '-subject:"adult" AND '
                '-subject:"erotica" AND '
                '-subject:"xxx" AND '
                f'downloads:[{min_views} TO 999999999]'
            )

            params = {
                "q": query,
                "output": "json",
                "rows": limit,
                "sort": "downloads desc",
                "fl": "identifier,title,description,year,downloads,num_reviews,avg_rating,subject"
            }

            response = requests.get(url, params=params)
            data = response.json()
            docs = data.get("response", {}).get("docs", [])

            # Filter and enrich results
            curated_movies = []
            for d in docs:
                # Additional content filtering
                subjects = d.get("subject", [])
                if isinstance(subjects, str):
                    subjects = [subjects]

                # Skip if contains adult-related keywords
                adult_keywords = ['adult', 'xxx', 'erotica', 'pornography', 'erotic']
                if any(keyword in ' '.join(subjects).lower() for keyword in adult_keywords):
                    continue

                downloads = d.get("downloads", 0)
                num_reviews = d.get("num_reviews", 0)
                avg_rating = d.get("avg_rating", 0)

                # Calculate quality score
                quality_score = (downloads * 0.5) + (num_reviews * 100) + (avg_rating * 1000)

                identifier = d.get("identifier")

                movie = {
                    "identifier": identifier,
                    "title": d.get("title"),
                    "description": (d.get("description", "No description available")[:200] + "...") if d.get("description") else "No description available",
                    "year": d.get("year"),
                    "downloads": downloads,
                    "num_reviews": num_reviews,
                    "avg_rating": round(avg_rating, 1) if avg_rating else 0,
                    "subjects": subjects[:5],
                    "quality_score": int(quality_score),
                    "thumbnail": f"https://archive.org/services/img/{identifier}",
                    "watch_url": f"https://archive.org/details/{identifier}",
                    "embed_url": f"https://archive.org/embed/{identifier}",
                    "platform": "Internet Archive",
                    "platform_short": "IA"
                }
                curated_movies.append(movie)

            # Sort by quality score
            curated_movies.sort(key=lambda x: x['quality_score'], reverse=True)

            return curated_movies

        except Exception as e:
            return {"error": str(e)}

    def search_youtube(self, query, max_results=10):
        """Search YouTube for videos"""
        try:
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": max_results,
                "key": YOUTUBE_API_KEY
            }

            response = requests.get(url, params=params)
            data = response.json()

            if "error" in data:
                return {"error": data["error"]["message"]}

            videos = []
            for item in data.get("items", []):
                video_id = item["id"]["videoId"]
                snippet = item["snippet"]

                videos.append({
                    "identifier": video_id,
                    "video_id": video_id,
                    "title": snippet.get("title"),
                    "description": snippet.get("description", "")[:200] + "..." if snippet.get("description") else "No description available",
                    "channel": snippet.get("channelTitle"),
                    "published_at": snippet.get("publishedAt"),
                    "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url"),
                    "watch_url": f"https://www.youtube.com/watch?v={video_id}",
                    "embed_url": f"https://www.youtube.com/embed/{video_id}",
                    "platform": "YouTube",
                    "platform_short": "YT"
                })

            return videos
        except Exception as e:
            return {"error": str(e)}

    def handle_search(self, parsed_path):
        """Handle search API request"""
        params = parse_qs(parsed_path.query)
        query = params.get('q', [''])[0]

        if not query:
            self.send_error(400, "Missing query parameter")
            return

        try:
            results = self.search_archive(query)

            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(results).encode())

        except Exception as e:
            self.send_error(500, f"Server error: {str(e)}")

    def handle_item(self, parsed_path):
        """Handle item details API request"""
        params = parse_qs(parsed_path.query)
        identifier = params.get('id', [''])[0]

        if not identifier:
            self.send_error(400, "Missing identifier parameter")
            return

        try:
            result = self.get_item_details(identifier)

            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        except Exception as e:
            self.send_error(500, f"Server error: {str(e)}")

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, IAHandler)
    print(f'🚀 AI-Powered Server running at http://localhost:{port}')
    print(f'🤖 ChatGPT integration enabled!')
    print(f'📱 Open http://localhost:{port}/index.html in your browser')
    print('Press Ctrl+C to stop the server')
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
