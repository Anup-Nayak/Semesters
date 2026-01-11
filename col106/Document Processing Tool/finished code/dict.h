// Do NOT add any other includes
#include <string> 
#include <vector>
#include <iostream>
#include <fstream>
using namespace std;


class trie_node{
public:
    trie_node* par;
    int word_count;
    vector<trie_node*> child=vector<trie_node*>(54,nullptr);
    trie_node();
    ~trie_node();
};

struct pair_util{
    trie_node* node;
    string word;
};


class trie{
public:
    trie_node* root;
    trie();
    ~trie();
    int search(string& word);
};

class Dict {
private:
    string separators = " .,-:!\"\'()?[];@";
    bool is_separator(char x);
    vector<pair_util> distinct_words;
    trie t;

public: 
    /* Please do not touch the attributes and 
    functions within the guard lines placed below  */
    /* ------------------------------------------- */
    Dict();

    ~Dict();

    void insert_sentence(int book_code, int page, int paragraph, int sentence_no, string sentence);

    int get_word_count(string word);

    void dump_dictionary(string filename);

    /* -----------------------------------------*/
};