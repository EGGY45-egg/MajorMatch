# MajorMatch Prompt Examples

Use these prompts to test each part of the app flow.

## 1. Landing and Orientation
- What can MajorMatch help me with?
- How does this app work?
- What do you do?

## 2. Chat Entry
- I like technology and problem solving. What should I explore?
- I’m good at coding and math. Can you help me choose a path?
- I’m not sure what I want to study yet.
  
(Notes: The first two prompts are likely to trigger the prediction flow if the user requests a recommendation; the system will request the front-end prediction UI or ask for selected features. The third is a general exploratory question and can be handled as normal chat unless the user asks for a recommendation.)

## 3. Career Track Prediction
- I’m strong in coding, math, and logic. What career should I pursue? (Should trigger `predict_track` — recommendation requested; assistant may ask to open the prediction UI.)
- Based on my skills, what major fits me best? (Should trigger `predict_track`.)
- Can you recommend a career track for someone who likes programming? (Should trigger `predict_track`.)

## 4. Career Context
- What is the salary outlook for software engineer?
- How many jobs are there for data scientist?
- Is UX design in demand right now?

(Notes: These prompts should trigger `get_career_context` to fetch live market data.)

## 5. Course Exploration
- Show me courses for web development.
- What classes should I take to become a data scientist?
- Find relevant courses for computer science.

(Notes: These should trigger `execute_semantic_search` to find and return course matches and a projection when requested.)

## 6. Visual Exploration
- Show me the course map for web development.
- Can you visualize the relevant courses?
- Plot the semantic search results for machine learning.

## 7. Refinement Loop
- That sounds good, but can you suggest another option?
- What if I’m stronger in design than coding?
- Show me a different path with higher salary potential.

(Notes: First two may trigger another `predict_track` run if a new recommendation is requested; the salary question could trigger `get_career_context` if phrased about market data.)

## 8. Actionable Outcome
- Give me a final recommendation based on my skills.
- Summarize the best major, salary, and courses for me.
- What should I do next if I want to become a software engineer?

(Notes: The first prompt should trigger `predict_track`. The second may combine outputs from `predict_track`, `get_career_context`, and `execute_semantic_search` depending on what the assistant needs to include.)
