// Main event listener for when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
  const pathname = window.location.pathname;

  // Route to the appropriate initialization function based on the current page
  if (pathname.endsWith("index.html")) {
    initializeChatPage();
  } else if (pathname.endsWith("static/ab_test.html")) {
    initializeABTestPage();
  }
});

// Initialize the chat page functionality
function initializeChatPage() {
  // Select necessary DOM elements
  const msgerForm = document.querySelector(".msger-inputarea");
  const msgerInput = document.querySelector(".msger-input");
  const msgerChat = document.querySelector(".msger-chat");
  const botNameSelect = document.querySelector("#bot-name-select");
  const userNameSelect = document.querySelector("#user-name-select");

  // Initialize variables
  let BOT_NAME, PERSON_NAME;
  const USE_API = false; // Set this to true when API is ready
  const PERSON_IMG = "./static/images/user.png";

  // Predefined bot messages for local response generation
  const BOT_MSGS = [
    "Hi, how are you?",
    "Ohh... I can't understand what you're trying to say. Sorry!",
    "I like to play games... But I don't know how to play!",
    "Sorry if my answers are not relevant. :))",
    "I feel sleepy! :("
  ];

  // Start the initialization process
  fetchBotNames();
  setupEventListeners();

  // Fetch bot names from JSON file and populate select options
  function fetchBotNames() {
    fetch('./static/botNames.json')
      .then(response => response.json())
      .then(data => {
        populateSelectOptions(data.botNames);
        BOT_NAME = botNameSelect.value;
        PERSON_NAME = userNameSelect.value;
      })
      .catch(error => console.error('Error fetching bot names:', error));
  }

  // Populate select options with fetched names
  function populateSelectOptions(names) {
    names.forEach(name => {
      userNameSelect.add(new Option(name, name));
      botNameSelect.add(new Option(name, name));
    });
  }

  // Set up event listeners for form submission and name changes
  function setupEventListeners() {
    botNameSelect.addEventListener('change', () => BOT_NAME = botNameSelect.value);
    userNameSelect.addEventListener('change', () => PERSON_NAME = userNameSelect.value);
    msgerForm.addEventListener("submit", handleSubmit);
  }

  // Handle form submission for sending a message
  function handleSubmit(event) {
    event.preventDefault();
    const msgText = msgerInput.value.trim();
    if (!msgText) return;

    appendMessage(PERSON_NAME, PERSON_IMG, "right", msgText);
    msgerInput.value = "";
    handleBotResponse(msgText);
  }

  // Generate and append bot's response
  function handleBotResponse(userMessage) {
    const BOT_IMG = `./static/images/bot_profile/${BOT_NAME}.png`;
    const botMessage = USE_API ?
      fetchBotResponseFromAPI(userMessage) :
      getLocalBotResponse();

    Promise.resolve(botMessage).then(msg =>
      appendMessage(BOT_NAME, BOT_IMG, "left", msg)
    );
  }

  // Generate a local bot response (used when API is not enabled)
  function getLocalBotResponse() {
    const randomIndex = Math.floor(Math.random() * BOT_MSGS.length);
    return `I am ${BOT_NAME}. You are ${PERSON_NAME}. ${BOT_MSGS[randomIndex]}`;
  }

  // Fetch bot response from API (when enabled)
  function fetchBotResponseFromAPI(userMessage) {
    return fetch('http://your-api-url/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: userMessage, botName: BOT_NAME })
    })
    .then(response => response.json())
    .then(data => data.reply)
    .catch(error => {
      console.error('Error fetching bot response:', error);
      return "Sorry, I couldn't connect to the server.";
    });
  }

  // Append a message to the chat interface
  function appendMessage(name, img, side, text) {
    const msgHTML = `
      <div class="msg ${side}-msg flex ${side === 'left' ? '' : 'justify-end'} mb-8">
        <div class="flex ${side === 'left' ? '' : 'flex-row-reverse'} max-w-[80%]">
          <div class="msg-img w-12 h-12 rounded-full bg-cover bg-center flex-shrink-0 ${side === 'left' ? 'mr-4' : 'ml-4'}" style="background-image: url(${img})"></div>
          <div class="msg-bubble ${side === 'left' ? 'bg-gray-200' : 'bg-blue-500 text-white'} rounded-lg p-4 shadow-md">
            <div class="msg-info flex justify-between mb-2">
              <div class="msg-info-name font-semibold">${name}</div>
              <div class="msg-info-time text-xs opacity-70 ml-4">${formatDate(new Date())}</div>
            </div>
            <div class="msg-text leading-relaxed">${text}</div>
          </div>
        </div>
      </div>
    `;

    msgerChat.insertAdjacentHTML("beforeend", msgHTML);
    msgerChat.scrollTop = msgerChat.scrollHeight;
  }

  // Format the current date for message timestamps
  function formatDate(date) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
}

// Initialize the A/B testing page functionality
function initializeABTestPage() {
  const characterSelect = document.getElementById("character_name");
  const votes = { "Model A": 0, "Model B": 0 };

  fetchBotNames();
  setupEventListeners();

  // Fetch bot names and populate the character select dropdown
  function fetchBotNames() {
    fetch('./botNames.json')
      .then(response => response.json())
      .then(data => {
        data.botNames.forEach(name => {
          characterSelect.add(new Option(name, name));
        });
      })
      .catch(error => console.error('Error fetching bot names:', error));
  }

  // Set up event listeners for submit and vote buttons
  function setupEventListeners() {
    document.getElementById("submit_btn").addEventListener("click", handleSubmit);
    document.getElementById("vote_a_btn").addEventListener("click", () => handleVote("Model A"));
    document.getElementById("vote_b_btn").addEventListener("click", () => handleVote("Model B"));
  }

  // Handle submission of user input for A/B testing
  function handleSubmit() {
    const inputText = document.getElementById("input_text").value;
    const characterName = characterSelect.value;

    document.getElementById("output_a").value = chatbotModel("A", inputText, characterName);
    document.getElementById("output_b").value = chatbotModel("B", inputText, characterName);
  }

  // Generate responses for both chatbot models
  function chatbotModel(model, inputText, characterName) {
    return `${characterName} (Model ${model}) responds: ${inputText}`;
  }

  // Handle voting for a particular model
  function handleVote(model) {
    votes[model]++;
    const voteMessage = `You voted for ${model}! Total votes: ${votes[model]}`;

    const messageBox = document.getElementById(`vote_${model.toLowerCase()}_message`);
    messageBox.value = voteMessage;
    messageBox.classList.remove("hidden");
  }
}
