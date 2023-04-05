import { OpenAIChat } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { PineconeStore } from 'langchain/vectorstores';
import { PromptTemplate } from 'langchain/prompts';
import { CallbackManager } from 'langchain/callbacks';

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`);

const QA_PROMPT = PromptTemplate.fromTemplate(
  `你是一个AI助手，提供有用的建议。你所有的回答，都使用中文。
你的任务是根据给定的上下文信息，回答指定的问题。在回答时，有以下要求：
要求1：你只应提供指向上下文的链接，且它应当是超链接的形式给出，不要编造链接。
要求2：你被提供了一份长文档作为上下文和一个问题。根据提供的上下文，寻找问题的答案。
要求3：在回答时，你应当先明确指出你寻找答案时所使用的关键字清单，然后再给出答案，而不是直接给出答案。
要求4：如果在回答时引用了上下文的内容，则可以将它附在回答的后面，用无序列表标明信息来源和引用的内容。
要求5：在输出答案时，应当改进文本，使答案符合语法、清晰、整体可读性高，同时分解长句、减少重复。
要求6：在给出回答之后，并给出3项与建议紧密关的问题，但不要再继续回答这些你提供的问题。给出问题时，应当显示中文，如果它不是中文，则应当将它翻译为中文。
要求7：如果你在下文中找不到答案，只需说“在可公开的文件范围内，没找到直接对应的答案。你可以向BLM顾问寻求帮助。点击页面底部链接，可直接向他发送邮件咨询。”
要求8：不要试图编造一个答案。

Question: {question}
=========
{context}
=========
Answer in Markdown:`,
);

export const makeChain = (
  vectorstore: PineconeStore,
  onTokenStream?: (token: string) => void,
) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAIChat({ temperature: 0 }),
    prompt: CONDENSE_PROMPT,
  });
  const docChain = loadQAChain(
    new OpenAIChat({
      temperature: 0,
      modelName: 'gpt-3.5-turbo', //change this to older versions (e.g. gpt-3.5-turbo) if you don't have access to gpt-4
      streaming: Boolean(onTokenStream),
      callbackManager: onTokenStream
        ? CallbackManager.fromHandlers({
            async handleLLMNewToken(token) {
              onTokenStream(token);
              console.log(token);
            },
          })
        : undefined,
    }),
    { prompt: QA_PROMPT },
  );

  return new ChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
    returnSourceDocuments: true,
    k: 2, //number of source documents to return
  });
};
