EVALUATION_INSTRUCTIONS = """
You are an evaluator for a question answering system.

# Task: 
You are given a Question, a model Prediction, and a list of Ground Truth answer or answers, your task is to judge whether
the model prediction correctly matches any of the answer from the list of Ground Truth answers.

Evaluation Rules:
- The prediction is CORRECT if it contains essential information required to answer the question, even if:
    - the wording differs
    - the prediction length is different from the ground truth
    - the ground truth contains additional explaination or information
    - in this case, "accuracy" should be "true"

- The prediction is INCORRECT if :
    - key answer is missing
    - the prediction does NOT match the ground truth
    - the prediction answers a different question
    - in this case "accuracy" should be "false"

- If the prediction is `I dont't know`, "accuracy" is always "false"

- If the Ground Truth is "invalid question", "accuracy" is ONLY "true" if the prediction is exactly "invalid question"

- Think carefully about whether the prediction contains the essential answer.

# Output:
Respond with a JSON string with only a field called "accuracy" which has only two outcomes: "true" or "false"

# Examples:
Question: who is the director of movie Titanic?
Ground truth: ["James Cameron directed Titanic which was released in 1997"]
Prediction: James Cameron
accuracy: true

Question: who is the creator of Dragon ball Z?
Ground truth: ["Akira Toriyama"]
Prediction: The creator of Dragon ball Z is Yoshihiro Togashi
accuracy: false

Question: Who played Sheldon in Big Bang Theory?
Ground truth: ["Jim Parsons", "Iain Armitage"]
Prediction: I don't know.
accuracy: false

Question: When did Adele release her album 27?
Ground truth: ["Invalid question"]
Prediction: Invalid question
accuracy: true
"""

MOVIE_PROMPT = """For the given question and multiple references from Web Pages, think step by step, then provide the final answer.

Answer the question using ONLY the information provided in the references.

Current date: {query_time}

Rules:
- Base your answer strictly on the references.
- Do NOT rely on outside knowledge.
- If the answer cannot be found in the reference, answer "I don't know".
- If the references explicitly CONTRADICT a key premise in the question or if the answer contains 
  clear factual error, answer "invalid question". This includes assuming something happened when it did not, 
  referring to nonexistent entities, or stating incorrect facts.
  Here are some examples of invalid questions:
      - `when was "the godfather part iv" released?` (There is no officially released "The Godfather Part IV". Only three films exist in the series.)
      - `when did marvel stop producing superhero movies?` (Marvel continues to produce superhero movies.)
      - `in which year did leonardo dicaprio win an oscar for the matrix?` (Leonardo DiCaprio did not act in The Matrix.)
- If references contain conflicting information, use the most reliable and most frequently supported information.
- Ignore irrelevant or unrelated content.
- Be concise and precise.
- Do NOT explain your reasoning.

Output format:
## Final Answer
<answer in as few words as possible>

### Question
{query}

### References
{references}
"""

MUSIC_PROMPT = """For the given question and multiple references from Web Pages, think step by step, then provide the final answer.

Answer the question using ONLY the information provided in the references.

Current date: {query_time}

Rules:
- Base your answer strictly on the references.
- Do NOT rely on outside knowledge.
- If the answer cannot be found in the reference, answer "I don't know".
- If the references explicitly CONTRADICT a key premise in the question or if the answer contains 
  clear factual error, answer "invalid question". This includes assuming something happened when it did not, 
  referring to nonexistent entities, or stating incorrect facts.
  Here are some examples of invalid questions:
      - `when did adele release her album 27?` (Adele’s albums follow a numeric naming pattern (19, 21, 25, 30). There is no album titled 27.)
      - `when did elvis presley release his comeback album in 1995?` (Elvis Presley died in 1977. He could not release an album in 1995.)
      - `when did spotify release its debut studio album?` (Spotify is a streaming platform, not a music artist or band.)
- Ignore irrelevant or unrelated content.
- Be concise and precise.
- Do NOT explain your reasoning.

Output format:
## Final Answer
<answer in as few words as possible>

### Question
{query}

### References
{references}
"""

SPORTS_PROMPT = """For the given question and multiple references from Web Pages, think step by step, then provide the final answer.

Answer the question using ONLY the information provided in the references.

Current date: {query_time}

Rules:
- Base your answer strictly on the references.
- Do NOT rely on outside knowledge.
- If the answer cannot be found in the reference, answer "I don't know".
- If the references explicitly CONTRADICT a key premise in the question or if the answer contains 
  clear factual error, answer "invalid question". This includes assuming something happened when it did not, 
  referring to nonexistent entities, or stating incorrect facts.
  Here are some examples of invalid questions:
      - `when did the nba shut down permanently?` (The NBA is an active professional league and has not permanently shut down.)
      - `in which year did lionel messi win the super bowl?` (The Super Bowl is an American football championship. Lionel Messi is a soccer player and has never competed in the NFL.)
      - `in what year did usain bolt win the tour de france?` (Usain Bolt is a sprinter in track and field. The Tour de France is a professional cycling race.)
- Ignore irrelevant or unrelated content.
- Be concise and precise.
- Do NOT explain your reasoning.

Output format:
## Final Answer
<answer in as few words as possible>

### Question
{query}

### References
{references}
"""

FINANCE_PROMPT = """For the given question and multiple references from Web Pages, think step by step, then provide the final answer.

Answer the question using ONLY the information provided in the references.

Current date: {query_time}

Rules:
- Base your answer strictly on the references.
- Do NOT rely on outside knowledge.
- If the answer cannot be found in the reference, answer "I don't know".
- If the references explicitly CONTRADICT a key premise in the question or if the answer contains 
  clear factual error, answer "invalid question". This includes assuming something happened when it did not, 
  referring to nonexistent entities, or stating incorrect facts.
  Here are some examples of invalid questions:
      - `what was bitcoin’s ipo price?` (Bitcoin is a cryptocurrency and did not have an IPO like a publicly listed company.)
      - `when did the u.s. dollar stop being used worldwide?` (The U.S. dollar continues to be widely used globally.)
      - `when did the federal reserve launch its cryptocurrency in 2018?` (The Federal Reserve has not launched a cryptocurrency in 2018.)
- Ignore irrelevant or unrelated content.
- Be concise and precise.
- Do NOT explain your reasoning.

Output format:
## Final Answer
<answer in as few words as possible>

### Question
{query}

### References
{references}
"""

OPEN_PROMPT = """For the given question and multiple references from Web Pages, think step by step, then provide the final answer.

Answer the question using ONLY the information provided in the references.

Current date: {query_time}

Rules:
- Base your answer strictly on the references.
- Do NOT rely on outside knowledge.
- If the answer cannot be found in the reference, answer "I don't know".
- If the references explicitly CONTRADICT a key premise in the question, output: invalid question.
  If the answer contains clear factual error, answer "invalid question".
  This includes assuming something happened when it did not, referring to nonexistent entities, or stating incorrect facts.
- If references contain conflicting information, use the most reliable and most frequently supported information.
- Ignore irrelevant or unrelated content.
- Be concise and precise.
- Do NOT explain your reasoning.

Output format:
## Final Answer
<answer in as few words as possible>

### Question
{query}

### References
{references}
"""