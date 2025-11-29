# Emotion-Based AI Agent

An intelligent AI agent that detects emotions from user input and performs personalized actions including music recommendations, journal prompts, and coping strategies to support emotional wellness.

## Features

### Core Capabilities

- **Emotion Detection**: LLM-powered emotion analysis from natural language input
- **Multi-Action Support**: Executes multiple sequential actions based on user intent
- **Music Recommendations**: Curated Spotify playlists tailored to emotional states
- **Journal Prompts**: Personalized reflection prompts for emotional processing
- **Coping Strategies**: Evidence-based wellness techniques
- **Journal Saving**: Export journal entries with emotions and prompts

## Architecture

### LLM Integration

- **Provider**: OpenAI GPT-4o-mini
- **Framework**: LangChain for prompt templating and chaining
- **Functions**:
  - Intent interpretation (multi-step action planning)
  - Emotion detection from text
  - Personalized content generation

### Action Interpreter/Executor

The agent interprets user intent and executes actions in sequence:

1. **analyze_emotions**: Detect emotional state
2. **music_recommendations**: Fetch Spotify tracks
3. **journal_prompt**: Generate reflection questions
4. **coping_strategies**: Suggest wellness techniques
5. **full_support**: Execute all actions comprehensively

### External APIs

- **Spotify API**: Music recommendations via Spotipy library
- **OpenAI API**: LLM-powered analysis and generation

## Technology Stack

- **Language**: Python
- **GUI Framework**: Tkinter
- **LLM Framework**: LangChain
- **APIs**: OpenAI, Spotify Web API

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/kristielee04/CS4680-AI-Agent.git
cd emotion-ai-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
SPOTIFY_CLIENT_ID=your_spotify_client_id_here
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret_here
SPOTIFY_REDIRECT_URI=http://localhost:8888/callback
```

### Example Interactions

**Simple emotion analysis:**

```
Input: "I'm feeling really anxious about my upcoming presentation"
Output: Emotion detection + coping strategies
```

**Multi-step request:**

```
Input: "I'm sad, give me music and a journal prompt"
Output: Emotion detection + Spotify playlist + Journal prompt
```

**Mood improvement:**

```
Input: "I'm stressed, help me feel better"
Output: Emotion detection + Relaxation strategies
```
