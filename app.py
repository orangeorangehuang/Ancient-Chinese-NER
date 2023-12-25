from run_ner_test import predict
import streamlit as st

class UI:
    def show(self):
        st.title("Ancient Chinese NER")
        st.subheader('An NER tool for ancient Chinese in traditional Chinese characters.', divider='rainbow')
        st.write("Named entities supported: personal names, locations, books, official titles, dynasties.")
        st.subheader("The Input Text")
        st.write("Recommended text length is around 300 words.")
        input_text = st.text_area("Please type in the text")

        if st.button("Submit", type="primary"):
            if input_text == "":
                st.error("The input text is required.")
                return
            else:
                with st.spinner("Processing..."):
                    text_result, label_result = predict(input_text)
                    st.subheader("Labelled Text")
                    st.write(text_result)
                    st.subheader("Labelled Tokens")
                    st.write(str(label_result).replace("'", ""))


if __name__ == "__main__":
    ui = UI()
    ui.show()
