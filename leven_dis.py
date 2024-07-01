from typing import List
import streamlit as st
from collections import defaultdict

def load_vocab(file_path: str) -> list:
    with open (file_path, 'r') as file:
        lines = file.readlines()
    words = sorted(set([line.strip().lower() for line in lines]))
    return words


def levenshtein_distance(token1: str, token2: str) -> int:
    len_token1 = len(token1)
    len_token2 = len(token2)

    # Create a distance matrix and initialize it
    dp: List[List[int]] = [[0] * (len_token1 + 1) for _ in range(len_token2 + 1)]

    for i in range(len_token2 + 1):
        dp[i][0] = i
    for j in range(len_token1 + 1):
        dp[0][j] = j

    # Compute the Levenshtein distance
    for i in range(1, len_token2 + 1):
        for j in range(1, len_token1 + 1):
            if token1[j - 1] == token2[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1,     # Deletion
                               dp[i][j - 1] + 1,     # Insertion
                               dp[i - 1][j - 1] + 1) # Substitution

    return dp[len_token2][len_token1]

def main():
    vocabs = load_vocab("./vocab.txt")
    
    st.title("Word correction using Levenshtein Distance")
    word = st.text_input("Word: ")

    if st.button("Compute"):
        level_distance = defaultdict(int)

        for vocab in vocabs:
            level_distance[vocab] = levenshtein_distance(word, vocab)
        
        sorted_distance = dict(
            sorted(level_distance.items(), key=lambda x: x[1])
        )
        correct_word = list(sorted_distance.keys())[0]

        st.write("Correct word: ", correct_word)
        col1, col2 = st.columns(2)
        col1.write('Vocabulary: ')
        col1.write(vocabs)

        col2.write("Distances: ")
        col2.write(sorted_distance)


if __name__ == "__main__":
    main()



