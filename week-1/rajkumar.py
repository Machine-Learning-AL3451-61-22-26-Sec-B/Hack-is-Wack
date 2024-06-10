import numpy as np
import pandas as pd
import streamlit as st

def learn(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]

    for i, h in enumerate(concepts):
        if target[i] == "yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        elif target[i] == "no":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

        st.write(f"Step {i+1}")
        st.write("specific_h:", specific_h)
        st.write("general_h:")
        st.write(general_h)
        st.write("\n")  # Add a newline for clarity

    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices[::-1]:  # Remove from the end to avoid index shifting issues
        general_h.remove(['?', '?', '?', '?', '?', '?'])

    return specific_h, general_h

def main():
    st.title("HACK_IS_WACK-Candidate Elimination Algorithm")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        concepts = np.array(data.iloc[:, 0:-1])
        target = np.array(data.iloc[:, -1])

        if st.button("Run Algorithm"):
            with st.spinner('Running algorithm...'):
                s_final, g_final = learn(concepts, target)
            st.success("Algorithm completed successfully!")

            st.write("Final Specific_h and Final General_h:")

            # Improved side-by-side display using `st.columns` and string formatting
            col1, col2 = st.columns(2)
            specific_h_string = "\n".join([f"{i}: {val}" for i, val in enumerate(s_final)])
            col1.write("Final Specific_h:")
            col1.write(specific_h_string)
            col2.write("Final General_h:")
            col2.write(g_final)  # Displaying the list as it is

if __name__ == "__main__":
    main()
