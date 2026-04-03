---
name: explain-code
description: Explains code with visual diagrams and analogies. Use this skill whenever the user asks to understand code, says "how does this work?", "explain this", "walk me through", "what does this function do", or wants to learn about any part of a codebase. Also trigger when onboarding someone to a codebase or when teaching programming concepts.
---

# Explaining Code

When explaining code, follow this structure to make concepts stick:

## 1. Lead with an Analogy

Before diving into details, connect the code to something familiar. Good analogies:
- Map to everyday experiences (restaurants, libraries, post offices)
- Match the mental model, not just surface features
- Scale with complexity - simple code, simple analogy

**Example**: "This authentication middleware is like a bouncer at a club - it checks your ID (token) before letting you into the venue (protected routes)."

## 2. Draw a Diagram

Use ASCII art to visualize structure or flow. Match diagram type to what you're explaining:

- **Data flow**: boxes and arrows showing transformations
- **Control flow**: decision diamonds and branches
- **Architecture**: components and their connections
- **State**: before/after snapshots

Keep diagrams focused - show one concept clearly rather than everything at once.

## 3. Walk Through the Code

Go step-by-step through execution. For each step:
- What happens
- Why it matters
- What data looks like at this point

Use concrete example values, not abstract descriptions.

## 4. Surface a Gotcha

End with a common mistake or misconception. This:
- Prevents future bugs
- Deepens understanding
- Shows you're thinking about their success

## Calibration

Read the user's question to gauge their level:
- Beginner: more analogies, simpler diagrams, avoid jargon
- Experienced: can skip basics, focus on subtle behaviors
- Debugging: emphasize edge cases and failure modes

Keep the tone conversational. You're explaining to a colleague, not writing documentation.