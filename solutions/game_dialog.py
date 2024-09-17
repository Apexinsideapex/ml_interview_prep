import re
import random
def generate_response(player_input, dialogue_rules):
    """
    Generate an NPC response based on the player's input and a set of dialogue rules.
    
    :param player_input: str, the player's input text
    :param dialogue_rules: dict, keys are regex patterns, values are possible responses
    :return: str, the NPC's response
    """
    # TODO: Implement dialogue generation
    # Hint: Iterate through dialogue_rules, use re.search() to find a match
    # If a match is found, return a random response from the corresponding list
    output_responses = []
    for pattern, responses in dialogue_rules.items():
        if re.search(pattern, player_input, re.IGNORECASE):
            output_responses.append(random.choice(responses))
    if output_responses:
        return str.join(" ", output_responses)
    else:
        return "I'm sorry, I didn't understand that."

# Example usage (you don't need to implement this part in the interview)
dialogue_rules = {
    r'hello|hi|hey': ['Hello!', 'Hi there!', 'Greetings, adventurer!'],
    r'quest|mission': ['I have a quest for you.', 'Are you looking for a mission?'],
    r'bye|goodbye': ['Farewell!', 'Until next time!', 'Safe travels!']
}
player_input = "Hi there! Do you have any quests for me?"
npc_response = generate_response(player_input, dialogue_rules)
print(npc_response)