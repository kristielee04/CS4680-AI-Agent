import os
import json
import logging
import webbrowser
from datetime import datetime
from dotenv import load_dotenv
import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading

# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Spotify
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_actions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmotionAgentGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion-Based AI Agent")
        self.root.geometry("1200x800")
        self.root.configure(bg="#ECF0F1")
        
        # Initialize agent backend
        self.initialize_agent()
        
        # Create GUI components
        self.create_widgets()
        
    def initialize_agent(self):
        """Initialize the AI agent backend"""
        try:
            # Initialize LangChain with OpenAI
            logger.info("Connecting to OpenAI...")
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found")
            
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.7,
                openai_api_key=api_key,
                request_timeout=30,
                max_retries=3
            )
            logger.info("OpenAI connected")
            
            # Initialize Spotify (keep as is)
            try:
                self.spotify = spotipy.Spotify(auth_manager=SpotifyOAuth(
                    client_id=os.getenv('SPOTIFY_CLIENT_ID'),
                    client_secret=os.getenv('SPOTIFY_CLIENT_SECRET'),
                    redirect_uri=os.getenv('SPOTIFY_REDIRECT_URI'),
                    scope="user-library-read"
                ))
                self.spotify_available = True
                logger.info("Spotify connected")
            except Exception as e:
                logger.warning(f"Spotify unavailable: {e}")
                self.spotify_available = False
                
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            messagebox.showerror("Initialization Error", f"Failed to initialize agent: {e}")
            self.root.quit()
    
    def create_widgets(self):
        """Create GUI components with two-column layout"""
        # Title bar
        title_frame = tk.Frame(self.root, bg="#2C3E50", height=75)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="Emotion-Based AI Agent",
            font=("Segoe UI", 20, "bold"),
            bg="#2C3E50",
            fg="white"
        )
        title_label.pack(side=tk.LEFT, padx=30, pady=20)
        
        subtitle_label = tk.Label(
            title_frame,
            text="Emotional Wellness Support System",
            font=("Segoe UI", 10),
            bg="#2C3E50",
            fg="#BDC3C7"
        )
        subtitle_label.pack(side=tk.LEFT, padx=(0, 30))
        
        # Main container with two columns
        main_container = tk.Frame(self.root, bg="#ECF0F1")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # LEFT PANEL - Input and Features
        left_panel = tk.Frame(main_container, bg="white", width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=0, pady=0)
        left_panel.pack_propagate(False)
        
        # Input section
        input_container = tk.Frame(left_panel, bg="white")
        input_container.pack(fill=tk.BOTH, expand=True, padx=25, pady=25)
        
        input_header = tk.Label(
            input_container,
            text="Share Your Feelings",
            font=("Segoe UI", 14, "bold"),
            bg="white",
            fg="#2C3E50"
        )
        input_header.pack(anchor=tk.W, pady=(0, 10))
        
        input_subtitle = tk.Label(
            input_container,
            text="Tell me how you're feeling or what you need help with",
            font=("Segoe UI", 9),
            bg="white",
            fg="#7F8C8D",
            wraplength=350,
            justify=tk.LEFT
        )
        input_subtitle.pack(anchor=tk.W, pady=(0, 15))
        
        self.input_text = tk.Text(
            input_container,
            height=6,
            font=("Segoe UI", 11),
            wrap=tk.WORD,
            relief=tk.SOLID,
            borderwidth=1,
            bg="#F8F9FA",
            padx=10,
            pady=10
        )
        self.input_text.pack(fill=tk.X, pady=(0, 15))
        
        # Buttons
        button_container = tk.Frame(input_container, bg="white")
        button_container.pack(fill=tk.X, pady=(0, 20))
        
        self.submit_button = tk.Button(
            button_container,
            text="Analyze & Get Support",
            command=self.process_input,
            font=("Segoe UI", 11, "bold"),
            bg="#3498DB",
            fg="white",
            cursor="hand2",
            relief=tk.FLAT,
            padx=20,
            pady=12,
            activebackground="#2980B9",
            activeforeground="white"
        )
        self.submit_button.pack(fill=tk.X, pady=(0, 10))
        
        self.clear_button = tk.Button(
            button_container,
            text="Clear",
            command=self.clear_all,
            font=("Segoe UI", 10),
            bg="#ECF0F1",
            fg="#2C3E50",
            cursor="hand2",
            relief=tk.FLAT,
            padx=20,
            pady=10,
            activebackground="#BDC3C7"
        )
        self.clear_button.pack(fill=tk.X)
        
        # Features section
        features_header = tk.Label(
            input_container,
            text="Available Actions",
            font=("Segoe UI", 12, "bold"),
            bg="white",
            fg="#2C3E50"
        )
        features_header.pack(anchor=tk.W, pady=(20, 15))
        
        features = [
            ("Emotion Analysis", "Detect and understand your emotional state"),
            ("Music Recommendations", "Get personalized song suggestions"),
            ("Journal Prompts", "Receive thoughtful reflection prompts"),
            ("Coping Strategies", "Learn wellness techniques")
        ]
        
        for feature_title, feature_desc in features:
            feature_frame = tk.Frame(input_container, bg="white")
            feature_frame.pack(fill=tk.X, pady=(0, 12))
            
            bullet = tk.Label(
                feature_frame,
                text="•",
                font=("Segoe UI", 12, "bold"),
                bg="white",
                fg="#3498DB"
            )
            bullet.pack(side=tk.LEFT, padx=(0, 8))
            
            text_frame = tk.Frame(feature_frame, bg="white")
            text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            title_label = tk.Label(
                text_frame,
                text=feature_title,
                font=("Segoe UI", 10, "bold"),
                bg="white",
                fg="#2C3E50",
                anchor=tk.W
            )
            title_label.pack(anchor=tk.W)
            
            desc_label = tk.Label(
                text_frame,
                text=feature_desc,
                font=("Segoe UI", 8),
                bg="white",
                fg="#7F8C8D",
                anchor=tk.W,
                wraplength=300,
                justify=tk.LEFT
            )
            desc_label.pack(anchor=tk.W)
        
        # RIGHT PANEL - Output
        right_panel = tk.Frame(main_container, bg="#F8F9FA")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        output_container = tk.Frame(right_panel, bg="#F8F9FA")
        output_container.pack(fill=tk.BOTH, expand=True, padx=25, pady=25)
        
        output_header_frame = tk.Frame(output_container, bg="#F8F9FA")
        output_header_frame.pack(fill=tk.X, pady=(0, 15))
        
        output_header = tk.Label(
            output_header_frame,
            text="Analysis Results",
            font=("Segoe UI", 14, "bold"),
            bg="#F8F9FA",
            fg="#2C3E50"
        )
        output_header.pack(side=tk.LEFT)
        
        self.status_indicator = tk.Label(
            output_header_frame,
            text="Ready",
            font=("Segoe UI", 9),
            bg="#F8F9FA",
            fg="#27AE60"
        )
        self.status_indicator.pack(side=tk.RIGHT)
        
        self.output_text = scrolledtext.ScrolledText(
            output_container,
            font=("Segoe UI", 10),
            wrap=tk.WORD,
            relief=tk.FLAT,
            borderwidth=0,
            state=tk.DISABLED,
            cursor="arrow",
            bg="white",
            padx=15,
            pady=15
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags
        self.output_text.tag_config("link", foreground="#3498DB", underline=True)
        self.output_text.tag_config("header", font=("Segoe UI", 11, "bold"), foreground="#2C3E50")
        self.output_text.tag_config("section", font=("Segoe UI", 10, "bold"), foreground="#34495E")
        self.output_text.tag_config("bold", font=("Segoe UI", 10, "bold"))

    def append_output_with_markdown(self, text):
        """Append text with markdown bold parsing"""
        self.output_text.config(state=tk.NORMAL)
        
        import re
        # Split text by **bold** markers
        parts = re.split(r'(\*\*.*?\*\*)', text)
        
        for part in parts:
            if part.startswith('**') and part.endswith('**'):
                # Remove ** and insert as bold
                bold_text = part[2:-2]
                self.output_text.insert(tk.END, bold_text, "bold")
            else:
                # Regular text
                self.output_text.insert(tk.END, part)
        
        self.output_text.insert(tk.END, "\n")
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)
        self.root.update_idletasks()
    
    def update_status(self, message):
        """Update status indicator with color coding"""
        self.status_indicator.config(text=message)
        if "Processing" in message or "Analyzing" in message or "Generating" in message or "Finding" in message:
            self.status_indicator.config(fg="#F39C12")
        elif "Complete" in message or "Ready" in message:
            self.status_indicator.config(fg="#27AE60")
        else:
            self.status_indicator.config(fg="#3498DB")
        self.root.update_idletasks()
    
    def append_output(self, text, tag=None):
        """Append text to output area"""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, text + "\n")
        if tag:
            start_idx = self.output_text.index(f"end-{len(text)+2}c")
            end_idx = self.output_text.index("end-1c")
            self.output_text.tag_add(tag, start_idx, end_idx)
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)
        self.root.update_idletasks()
    
    def add_hyperlink(self, display_text, url):
        """Helper function to insert text and bind a hyperlink tag"""
        tag_name = f"link-{url}"
        self.output_text.insert(tk.END, display_text, ("link", tag_name))
        self.output_text.tag_configure("link", foreground="#3498DB", underline=True)
        self.output_text.tag_bind(tag_name, "<Button-1>", lambda e: webbrowser.open_new_tab(url))
        self.output_text.tag_bind(tag_name, "<Enter>", lambda e: self.output_text.config(cursor="hand2"))
        self.output_text.tag_bind(tag_name, "<Leave>", lambda e: self.output_text.config(cursor="arrow"))
    
    def clear_all(self):
        """Clear all text fields"""
        self.input_text.delete(1.0, tk.END)
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state=tk.DISABLED)
        self.update_status("Ready")
    
    def process_input(self):
        """Process user input in a separate thread"""
        user_input = self.input_text.get(1.0, tk.END).strip()
        
        if not user_input:
            messagebox.showwarning("Empty Input", "Please share how you're feeling!")
            return
        
        self.submit_button.config(state=tk.DISABLED)
        
        thread = threading.Thread(target=self.run_agent, args=(user_input,))
        thread.daemon = True
        thread.start()
    
    def run_agent(self, user_input):
        """Run agent logic"""
        try:
            logger.info(f"Processing input: '{user_input[:50]}...'")
            
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete(1.0, tk.END)
            self.output_text.config(state=tk.DISABLED)
            
            self.update_status("Understanding your request...")
            self.append_output("Understanding your request...\n", "header")
            actions, reasoning = self.interpret_user_intent(user_input)
            self.append_output(f"-> {reasoning}")
            
            if len(actions) > 1:
                self.append_output(f"-> I'll complete {len(actions)} steps for you:\n")
                for i, action in enumerate(actions, 1):
                    action_name = action.replace('_', ' ').title()
                    self.append_output(f"   {i}. {action_name}")
            self.append_output("")
            
            self.update_status("Analyzing emotions...")
            self.append_output("Analyzing your emotions...\n", "header")
            emotions = self.detect_emotions(user_input)

            if not emotions:
                self.append_output("Couldn't detect clear emotions. Try being more descriptive!\n")
                self.submit_button.config(state=tk.NORMAL)
                self.update_status("Ready")
                return

            self.append_output(f"{emotions}\n")
            
            journal_prompt = None
            
            for step_num, action in enumerate(actions, 1):
                if len(actions) > 1:
                    self.append_output(f"\n--- Step {step_num}/{len(actions)}: {action.replace('_', ' ').title()} ---\n")
                
                if action == "analyze_emotions":
                    self.append_output("Emotion analysis complete!\n")
                    
                elif action == "music_recommendations":
                    if self.spotify_available:
                        self.update_status("Finding music...")
                        self.append_output("Finding music for your mood...\n", "header")
                        spotify_recs = self.get_spotify_recommendations(emotions, user_input)
                        
                        if spotify_recs:
                            mood_desc = spotify_recs['mood_description']
                            self.append_output(f"Here are {mood_desc}\n")
                            self.append_output("Recommended Songs:\n", "section")
                            for i, track in enumerate(spotify_recs['tracks'][:5], 1):
                                self.output_text.config(state=tk.NORMAL)
                                self.output_text.insert(tk.END, f"   {i}. {track['name']} - {track['artist']}\n      ")
                                self.add_hyperlink(track['url'], track['url'])
                                self.output_text.insert(tk.END, "\n")
                                self.output_text.config(state=tk.DISABLED)
                                self.output_text.see(tk.END)
                    else:
                        self.append_output("Music recommendations unavailable (Spotify not connected)\n")
                    
                elif action == "journal_prompt":
                    self.update_status("Generating journal prompt...")
                    self.append_output("Generating your journal prompt...\n", "header")
                    journal_prompt = self.generate_journal_prompt(emotions, user_input)
                    self.append_output(f"Journal Prompt:\n{journal_prompt}\n")
                
                elif action == "coping_strategies":
                    self.update_status("Generating coping strategies...")
                    self.append_output("Suggesting coping strategies...\n", "header")
                    strategies = self.generate_coping_strategies(emotions, user_input)
                    self.append_output_with_markdown(strategies)
                        
                elif action == "full_support":
                    if self.spotify_available:
                        self.update_status("Finding music...")
                        self.append_output("Finding music for your mood...\n", "header")
                        spotify_recs = self.get_spotify_recommendations(emotions, user_input)
                        
                        if spotify_recs:
                            mood_desc = spotify_recs['mood_description']
                            self.append_output(f"Here are {mood_desc}\n")
                            self.append_output("Recommended Songs:\n", "section")
                            for i, track in enumerate(spotify_recs['tracks'][:5], 1):
                                self.output_text.config(state=tk.NORMAL)
                                self.output_text.insert(tk.END, f"   {i}. {track['name']} - {track['artist']}\n      ")
                                self.add_hyperlink(track['url'], track['url'])
                                self.output_text.insert(tk.END, "\n")
                                self.output_text.config(state=tk.DISABLED)
                                self.output_text.see(tk.END)
                    else:
                        self.append_output("Music recommendations unavailable (Spotify not connected)\n")

                    self.update_status("\nGenerating journal prompt...")
                    self.append_output("Generating personalized journal prompt...\n", "header")
                    journal_prompt = self.generate_journal_prompt(emotions, user_input)
                    self.append_output(f"Journal Prompt:\n{journal_prompt}\n")
            
            if journal_prompt:
                self.root.after(0, lambda: self.offer_save(user_input, emotions, journal_prompt))
            
            self.update_status("Complete!")
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            self.append_output(f"\nError: {e}\n")
            messagebox.showerror("Error", f"An error occurred: {e}")
        
        finally:
            self.submit_button.config(state=tk.NORMAL)
    
    def offer_save(self, user_input, emotions, journal_prompt):
        """Ask user if they want to save journal entry"""
        response = messagebox.askyesno(
            "Save Journal Entry",
            "Would you like to save this journal prompt to a file?"
        )
        
        if response:
            filename = self.save_journal_entry(user_input, emotions, journal_prompt)
            if filename:
                messagebox.showinfo("Saved", f"Journal entry saved to:\n{filename}")
                logger.info(f"Journal saved: {filename}")
            else:
                messagebox.showerror("Error", "Failed to save journal entry")
    
    def interpret_user_intent(self, user_input):
        """Interpret user intent using LLM"""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an AI agent that interprets user requests related to emotions and mental wellness.
                Analyze the user's input and determine what actions they want you to take.
                
                Available actions (can be combined for multi-step tasks):
                1. "analyze_emotions" - Detect emotions from their feelings/day
                2. "journal_prompt" - Generate a journal writing prompt
                3. "music_recommendations" - Suggest music for their mood
                4. "coping_strategies" - Suggest coping strategies for their emotional state
                5. "full_support" - Do all of the above (default for emotional sharing)
                
                For multi-step tasks, return multiple actions in order.
                
                Respond with ONLY a JSON object (no markdown):
                {{"actions": ["action1", "action2"], "reasoning": "brief explanation"}}
                
                Examples:
                - "I'm stressed" -> {{"actions": ["full_support"], "reasoning": "User needs comprehensive support"}}
                - "Create a playlist and journal prompt" -> {{"actions": ["music_recommendations", "journal_prompt"], "reasoning": "Multi-step request for music and journaling"}}
                - "Analyze my mood then give me coping strategies" -> {{"actions": ["analyze_emotions", "coping_strategies"], "reasoning": "Sequential emotion analysis and coping advice"}}
                - "Just give me music" -> {{"actions": ["music_recommendations"], "reasoning": "Direct music request"}}
                """),
                ("user", "{user_input}")
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            llm_response = chain.invoke({"user_input": user_input})
            
            import re
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                intent_data = json.loads(json_match.group())
            else:
                intent_data = json.loads(llm_response)
            
            actions = intent_data.get('actions', ['full_support'])
            reasoning = intent_data.get('reasoning', 'Processing user request')
            
            logger.info(f"Intent: {actions} - {reasoning}")
            return actions, reasoning
            
        except Exception as e:
            logger.error(f"Error interpreting intent: {e}")
            return ["full_support"], "Processing emotional state"
    
    def detect_emotions(self, text):
        """Detect emotions using LLM and return human-readable response"""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert emotion detection AI. Analyze the user's text and identify the emotions present.
                
                Respond with a natural sentence that lists the detected emotions (1-3 emotions).
                Format: "I have detected the following emotions based on your input: [emotion1], [emotion2], and [emotion3]."
                
                Use emotions like: joy, sadness, anger, fear, surprise, love, anxiety, excitement, frustration, contentment, worry, hope, etc.
                
                Be empathetic and natural in your phrasing.
                """),
                ("user", "Analyze emotions in this text: {text}")
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            emotion_response = chain.invoke({"text": text})
            
            logger.info(f"Detected emotions: {emotion_response}")
            return emotion_response
            
        except Exception as e:
            logger.error(f"Error detecting emotions: {e}")
            return "I have detected the following emotions based on your input: uncertainty."
    
    def generate_coping_strategies(self, emotions, user_input):
        """Generate coping strategies based on emotions"""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a mental wellness expert. Based on the user's emotions, 
                suggest 3-4 practical, evidence-based coping strategies they can use right now.
                Be specific, actionable, and empathetic. Keep each strategy to 1-2 sentences."""),
                ("user", """User's emotions: {emotions}
                User's input: {user_input}
                
                Suggest coping strategies:""")
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            strategies = chain.invoke({
                "emotions": emotions,
                "user_input": user_input
            })
            
            logger.info("Coping strategies generated")
            return strategies
            
        except Exception as e:
            logger.error(f"Error generating coping strategies: {e}")
            return "Take deep breaths, practice mindfulness, and consider talking to someone you trust."
    
    def generate_journal_prompt(self, emotions, user_input):
        """Generate journal prompt using LLM"""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an empathetic journaling assistant. Based on the user's 
                emotions and their input, generate a thoughtful, personalized journal prompt 
                that helps them reflect on their feelings. The prompt should be 2-3 sentences, 
                encouraging, and open-ended."""),
                ("user", """User's emotions: {emotions}
                User's input: {user_input}
                
                Generate a personalized journal prompt:""")
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            journal_prompt = chain.invoke({
                "emotions": emotions,
                "user_input": user_input
            })
            
            logger.info("Journal prompt generated")
            return journal_prompt
            
        except Exception as e:
            logger.error(f"Error generating journal prompt: {e}")
            return "Take a moment to reflect on your feelings. What thoughts are running through your mind?"
    
    def get_spotify_recommendations(self, emotions, user_input):
        """Get Spotify recommendations using LLM"""
        if not self.spotify_available:
            return None
        
        try:
            import random
            # Add randomness seed to prompt for variety
            variety_seed = random.randint(1, 1000)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a music recommendation expert. Based on the user's emotions AND their request, 
                suggest appropriate music. 
                
                IMPORTANT: Determine if the user wants to:
                1. MATCH their current mood (reflective, validating their feelings)
                2. IMPROVE their mood (uplifting, energizing, mood-boosting)
                
                Look for keywords like "feel better", "cheer me up", "happier", "lift my mood" → suggest UPLIFTING music
                Otherwise, match their current emotional state.
                
                For MAXIMUM VARIETY, use diverse and specific search terms:
                - Mix mainstream and indie artists
                - Use different subgenres each time (indie folk, synthpop, lo-fi, jazz fusion, etc.)
                - Include decade/era descriptors (90s, modern, classic)
                - Use descriptive mood words (dreamy, raw, vibrant, mellow)
                - Avoid repeating the same search patterns
                
                Respond with ONLY a JSON object (no markdown):
                {{"search_queries": ["query1", "query2", "query3"], "mood_description": "brief description"}}
                
                Examples:
                - Fear (high) + "help me relax" → {{"search_queries": ["ambient chillout downtempo", "lo-fi hip hop beats", "nature sounds instrumental"], "mood_description": "calming tracks to help you relax"}}
                - Sadness (medium) + "I'm sad" → {{"search_queries": ["melancholic indie folk", "emotional alternative ballads", "introspective singer songwriter"], "mood_description": "gentle, reflective songs that validate what you're feeling"}}
                - Sadness + "make me feel better" → {{"search_queries": ["uplifting indie pop", "feel good acoustic", "inspirational soul"], "mood_description": "uplifting tracks to help you feel better"}}
                - Joy (high) → {{"search_queries": ["upbeat indie dance", "tropical house summer", "feel good funk groove"], "mood_description": "energetic songs that celebrate your joy"}}
                
                Variety seed: {variety_seed} (use this to generate different search terms each time)
                """),
                ("user", "Emotions: {emotions}\nUser input: {user_input}\n\nGenerate music search queries:")
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            llm_response = chain.invoke({
                "emotions": emotions,
                "user_input": user_input,
                "variety_seed": variety_seed
            })
            
            import re
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                rec_data = json.loads(json_match.group())
            else:
                rec_data = json.loads(llm_response)
            
            search_queries = rec_data.get('search_queries', [])
            mood_description = rec_data.get('mood_description', 'your mood')
            intent = rec_data.get('intent', 'match_mood')
            
            all_tracks = []
            seen_artists = set()  # Avoid duplicate artists
            
            for query in search_queries[:3]:  # Use all 3 queries for more variety
                try:
                    # Randomize offset to get different results
                    offset = random.randint(0, 20)
                    results = self.spotify.search(q=query, type='track', limit=10, offset=offset)
                    
                    if results and 'tracks' in results:
                        for track in results['tracks']['items']:
                            if track and track['id'] not in [t['id'] for t in all_tracks]:
                                # Skip if we already have a song from this artist
                                artist_name = track['artists'][0]['name'] if track['artists'] else ''
                                if artist_name in seen_artists:
                                    continue
                                
                                seen_artists.add(artist_name)
                                artists = ', '.join([a['name'] for a in track['artists']])
                                all_tracks.append({
                                    'id': track['id'],
                                    'name': track['name'],
                                    'artist': artists,
                                    'url': track['external_urls']['spotify']
                                })
                                
                                if len(all_tracks) >= 10:
                                    break
                except Exception as e:
                    logger.warning(f"Search failed for '{query}': {e}")
                    continue
                
                if len(all_tracks) >= 10:
                    break
            
            # Shuffle the final list for extra randomness
            if all_tracks:
                random.shuffle(all_tracks)
                return {
                    'mood_description': mood_description, 
                    'tracks': all_tracks[:10],
                    'intent': intent
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting Spotify recommendations: {e}")
            return None
    
    def save_journal_entry(self, user_input, emotions, journal_prompt):
        """Save journal entry to file"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"journal_entry_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("="*50 + "\n")
                f.write(f"Journal Entry - {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n")
                f.write("="*50 + "\n\n")
                f.write(f"How I felt:\n{user_input}\n\n")
                f.write(f"Detected Emotions:\n{emotions}\n")
                f.write(f"\n{'='*50}\n")
                f.write(f"Journal Prompt:\n\n{journal_prompt}\n")
                f.write(f"\n{'='*50}\n")
                f.write("\nYour thoughts:\n\n\n\n")
            
            return filename
        except Exception as e:
            logger.error(f"Error saving journal: {e}")
            return None

def main():
    root = tk.Tk()
    EmotionAgentGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()