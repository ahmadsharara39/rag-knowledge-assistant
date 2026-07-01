# How the RAG Pipeline Works

Retrieval-Augmented Generation (RAG) grounds a language model's answers in your
own documents so responses are accurate and citable instead of relying only on
the model's training data.

The pipeline has five stages. First, documents are split into overlapping,
sentence-aware chunks so each unit is small enough to embed well while keeping
enough context. Second, every chunk is converted to an embedding. Third, the user's
question is embedded and the vector store returns the top-k most similar chunks via
semantic search. Fourth, those chunks are assembled into a grounded prompt with
numbered source markers. Fifth, the language model writes an answer using only that
context and cites each claim with a marker such as [1] or [2].

Because answers are grounded in retrieved context and every claim is cited, the
system is auditable: a reviewer can trace each statement back to its source chunk.
If the answer is not present in the retrieved context, the assistant is instructed
to say it does not have enough information rather than guess.
