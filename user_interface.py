import os
import random

import pandas as pd
import streamlit as st
from ContentBasedRS import ContentDB, ProfileDB, Recommender, OpenAIEmbedding

content_db = ContentDB()
profile_db = ProfileDB()
recommender = Recommender(content_db, profile_db)

# ---------------------------------------------cahced fp----------------------------------------------------------------
cached_recommendation_file_path = "ContentBasedRS/cache_function_call/cached_recommendation.pkl"
cached_author_name_list_file_path = "ContentBasedRS/cache_function_call/cached_author_name_list.pkl"
cached_advanced_recommendation_file_path = "ContentBasedRS/cache_function_call/cached_advanced_recommendation.pkl"


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------page 1-------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def tourist_mode():
    st.subheader("Tourist Mode")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Search Engine", "Latest", "Most Referenced", "Most Cited",
                                            "Most influential"])
    with tab1:
        _search_engine()
    with tab2:
        _order_by_field("latest", content_db.COL_YEAR)
    with tab3:
        _order_by_field("most referenced", content_db.COL_REF_COUNT)
    with tab4:
        _order_by_field("most cited", content_db.COL_CITE_COUNT)
    with tab5:
        _order_by_field("most influential", content_db.COL_INFLUENTIAL_CITE_COUNT)


def _search_engine():
    st.subheader("Search Engine")
    openai_api_key = st.text_input("Please enter your openAI API key: ", type="password")
    if openai_api_key:
        st.empty()
        os.environ["OPENAI_API_KEY"] = openai_api_key
        keyword = st.text_input("Please enter topics you are interested in: ")
        if keyword:
            recommendation = recommender.search_engine(keyword)
            recommendation = _polish_contentdb_df(recommendation)
            st.dataframe(recommendation)


def _order_by_field(category, order_by_column_name):
    button_clicked = st.button(f"Find me {category} papers!")
    if button_clicked:
        recommendation = recommender.order_by_number(order_by_column_name)
        recommendation = _polish_contentdb_df(recommendation)
        st.dataframe(recommendation)
        button_clicked = False


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------page 2-------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def sign_up():
    st.subheader("Sign up")
    user_name = st.text_input("Please enter your name")
    if user_name:
        user_id = _generate_random_id()
        st.write(f"your user id is {user_id}")
        # add at least one known paper
        known_papers = {profile_db.PAPER_KIND_WRITE: {"f9c602cc436a9ea2f9e7db48c77d924e09ce3c32"},
                        profile_db.PAPER_KIND_REF: set(),
                        profile_db.PAPER_KIND_LIKED: set()}

        profile_db.update_author(user_id, user_name, known_papers)
        profile_db.commit_change()


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------page 3-------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def be_an_author():
    st.subheader("I am an Author")
    tab1, tab2, tab3 = st.tabs(["Main Page", "Operations", "Author Information"])
    author_id = None
    with tab1:
        options = st.radio("Do you know what's the author's id?",
                           ("I already know my author id", "I don't know my author id..."))
        if options == "I don't know my author id...":
            author_id, recommendations = _recommend_given_author_name()
        else:
            author_id, recommendations = _recommend_given_author_id()
        # show boxes for the person to choose what paper he like
        if author_id is not None and recommendations is not None:
            _user_select_liked_papers(author_id, recommendations)
    with tab2:
        # todo: there might be a bug here: the database will lock if multy thread,
        #  but streamlit seems to have multithreading call
        if author_id is None:
            st.write("You haven't select an author id yet!")
        else:
            _clear_liked_papers_page(author_id)
            author_id = _delete_this_account(author_id)
    with tab3:
        if author_id is None:
            st.write("You haven't select an author id yet!")
        else:
            _author_info_page(author_id)


def _generate_random_id():
    while True:
        my_id = str(random.randint(0, 9999))
        result = profile_db.query_database(
            f"select * from {profile_db.MAIN_TABLE_NAME} where {profile_db.COL_AUTHOR_ID} = {my_id}")
        if result.empty:
            break
    return my_id


def _clear_liked_papers_page(author_id):
    button_clicked = st.button("Clear my liked papers!")
    if button_clicked:
        profile_db.clear_liked_papers(author_id)
        profile_db.commit_change()
        st.write("All liked papers is deleted!")
        button_clicked = False


def _delete_this_account(author_id):
    button_clicked = st.button("Delete this account!")
    st.write("Warning: Don't delete default authors!!!")
    if button_clicked:
        profile_db.delete_profile(author_id)
        profile_db.commit_change()
        st.write("This account is deleted!")
        button_clicked = False
        # clear cache in case show wrong information
        if os.path.exists(cached_author_name_list_file_path):
            os.remove(cached_author_name_list_file_path)
        if os.path.exists(cached_recommendation_file_path):
            os.remove(cached_recommendation_file_path)
        return None
    return author_id


def _author_info_page(author_id):
    author_info = profile_db.get_row_by_id(author_id)[[profile_db.COL_AUTHOR_ID, profile_db.COL_NAME]]
    st.write(f"The author's name and id is:")
    st.dataframe(author_info)
    known_papers = profile_db.get_author_known_papers(author_id)
    for paper_category, paper_id_set in known_papers.items():
        st.write(paper_category)
        known_papers_df = content_db.get_papers_by_id_set(paper_id_set)
        known_papers_df = _polish_contentdb_df(known_papers_df)
        st.dataframe(known_papers_df)


def _recommend_given_author_name():
    partial_author_name = st.text_input("Please write the author you want to be: ")
    button_clicked = st.button("Find authors that have this name! ")
    if partial_author_name and button_clicked:
        possible_authors = profile_db.search_author_by_name(partial_author_name)
        possible_authors.to_pickle(cached_author_name_list_file_path)
        st.dataframe(possible_authors)
        author_id, recommendations = _recommend_given_author_id()
        button_clicked = False
        return author_id, recommendations
    elif partial_author_name and os.path.exists(cached_author_name_list_file_path):
        # cached the result so when stream lit refresh the page it won't rerun the query
        possible_authors = pd.read_pickle(cached_author_name_list_file_path)
        st.dataframe(possible_authors)
        author_id, recommendations = _recommend_given_author_id()
        return author_id, recommendations
    # if some one clear the search box/change to another page, clear cache, so it won't show wrong information before
    # next search
    if os.path.exists(cached_author_name_list_file_path):
        os.remove(cached_author_name_list_file_path)
    return None, None


def _recommend_given_author_id():
    author_id = st.text_input("please input the author id you wish to be: ")

    # advance mode
    check_box_clicked = st.checkbox("Advance mode")

    if check_box_clicked:
        return _advance_recommend_mode(author_id)
    else:
        # todo: button clicked will update?
        return _normal_recommend_mode(author_id, recommender.recommend_to_author)


def _advance_recommend_mode(author_id):
    cos_weight = st.number_input("Cosine similarity weight: ", min_value=0.0, max_value=1.0, value=0.6)
    ref_weight = st.number_input("Referenced number weight: ", min_value=0.0, max_value=1.0, value=0.1)
    cite_weight = st.number_input("Cited number weight: ", min_value=0.0, max_value=1.0, value=0.1)
    inf_weight = st.number_input("Influential citation number weight: ", min_value=0.0, max_value=1.0, value=0.1)
    year_weight = st.number_input("Published year weight: ", min_value=0.0, max_value=1.0, value=0.1)
    # check sum
    total_sum = cos_weight + ref_weight + cite_weight + inf_weight + year_weight
    if abs(total_sum - 1) > 1e-5:
        st.warning("The sum of weights should sum up to 1")
    return _normal_recommend_mode(author_id, recommender.recommend_by_weighted_linear_model,
                                  cached_advanced_recommendation_file_path,
                                  cos_weight, ref_weight, cite_weight, inf_weight, year_weight)


def _normal_recommend_mode(author_id, recommender_function_callback, cache_path=cached_recommendation_file_path,
                           *callback_args):
    button_clicked = st.button("Find papers that would interest me! ")
    if author_id and button_clicked:
        # recommendation = recommender.recommend_to_author(author_id)
        recommendation = recommender_function_callback(author_id, *callback_args)
        st.write("These are papers that this author might be interested in: ")
        recommendation.to_pickle(cache_path)
        recommendation = _polish_contentdb_df(recommendation)
        st.dataframe(recommendation)
        button_clicked = False
        return author_id, recommendation
    elif author_id and os.path.exists(cache_path):
        # cached the result so when stream lit refresh the page it won't rerun the query
        recommendation = pd.read_pickle(cache_path)
        st.write("These are papers that this author might be interested in: ")
        recommendation = _polish_contentdb_df(recommendation)
        st.dataframe(recommendation)
        return author_id, recommendation
    # if some one clear the search box/change to another page, clear cache, so it won't show wrong information before
    # next search
    if os.path.exists(cache_path):
        os.remove(cache_path)

    return None, None


def _user_select_liked_papers(author_id, recommendations):
    # present all the options
    choose_from = recommendations[content_db.COL_TITLE].tolist()
    chosen_paper = st.multiselect("Please select papers that you like: ", choose_from)
    # get the liked papers (id)
    chosen_paper_df = recommendations.loc[recommendations[content_db.COL_TITLE].isin(chosen_paper)]
    liked_papers = set(chosen_paper_df[content_db.COL_PAPER_ID].tolist())
    # update this user's preference if asked
    button_clicked = st.button("Save my preference")
    if button_clicked:
        new_known_papers = {profile_db.PAPER_KIND_LIKED: liked_papers}
        # assume author id is real: already checked above
        author_name = profile_db.get_row_by_id(author_id)[profile_db.COL_NAME][0]
        profile_db.update_author(author_id, author_name, new_known_papers)
        profile_db.commit_change()
        st.write("Your preference is saved!")
        button_clicked = False


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------help function------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def _polish_contentdb_df(df):
    if content_db.COL_AUTHORS in df.columns and content_db.COL_REFERENCE in df.columns:
        df = df.apply(_change_authors_and_references, axis=1)
    return df


def _change_authors_and_references(row):
    authors = row[content_db.COL_AUTHORS]
    represent_authors = [author_info["name"] for author_info in authors]
    row[content_db.COL_AUTHORS] = represent_authors
    references = row[content_db.COL_REFERENCE]
    represent_references = [ref_info['paperId'] for ref_info in references]
    row[content_db.COL_REFERENCE] = represent_references
    return row


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------main function------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def main():
    st.title("Machine Learning Research Papers Recommender System")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Features:", ("Tourist Mode", "Sign Up", "I am an Author"))

    if page == "Tourist Mode":
        tourist_mode()
    if page == "Sign Up":
        sign_up()
    if page == "I am an Author":
        be_an_author()


if __name__ == "__main__":
    main()
    # todo: represent recommendation, dislike?????
