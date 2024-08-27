// Main event listener for when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
  const pathname = window.location.pathname;

  // Route to the appropriate initialization function based on the current page
  if (pathname.endsWith("ab_test")) {
    initializeABTestPage();
  }
  else {
    initializeChatPage();
  }
});

function getApiUrl() {
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    return 'http://localhost:8000';
  } else {
    return 'http://158.247.203.184:8000';
  }
}

// Storing the object
function storeObject(key, object) {
  sessionStorage.setItem(key, JSON.stringify(object));
}

// Retrieving the object
function getObject(key) {
  const item = sessionStorage.getItem(key);
  return item ? JSON.parse(item) : null;
}
// Usage example:
const API_URL = getApiUrl();
// Initialize the chat page functionality
function initializeChatPage() {
  // Select necessary DOM elements
  const msgerForm = document.querySelector(".msger-inputarea");
  const msgerInput = document.querySelector(".msger-input");
  const msgerChat = document.querySelector(".msger-chat");
  const botNameSelect = document.querySelector("#bot-name-select");
  const userNameSelect = document.querySelector("#user-name-select");

  const API_URL = getApiUrl(); // Use the new getApiUrl function

  // Initialize variables
  let BOT_NAME, PERSON_NAME;
  const USE_API = true; // Set this to true when API is ready
  const PERSON_IMG = "/client/static/images/user.png";

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
    fetch('/client/static/botNames.json')
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
    const BOT_IMG = `/client/static/images/bot_profile/${BOT_NAME}.png`;
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
    return fetch(`${API_URL}/invoke`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        llm: "openai",
        retriever: "metadata",
        rag: "base",
        character: BOT_NAME,
        prompt: userMessage
      })
    })
    .then(response => response.json())
    .then(data => data.generation) // Assuming the API returns the response in an 'output' field
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
  const API_URL = getApiUrl();

  fetchBotNames();
  setupEventListeners();

  // Fetch bot names and populate the character select dropdown
  function fetchBotNames() {
    fetch('/client/static/botNames.json')
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
    document.getElementById("vote_a_btn").addEventListener("click", () => handleVote("A"));
    document.getElementById("vote_b_btn").addEventListener("click", () => handleVote("B"));
  }

  function showSpinner() {
    document.getElementById("spinner").classList.remove("hidden");
    document.getElementById("submit_btn").disabled = true;
  }

  function hideSpinner() {
    document.getElementById("spinner").classList.add("hidden");
    document.getElementById("submit_btn").disabled = false;
  }

  // Handle submission of user input for A/B testing
  function handleSubmit() {
    const inputText = document.getElementById("input_text").value;
    const characterName = characterSelect.value;

    showSpinner();

    fetch(`${API_URL}/random`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        character: characterName,
        prompt: inputText
      })
    })
    .then(response => response.json())
    .then(data => {
      document.getElementById("output_a").value = data.A.output.generation;
      document.getElementById("output_b").value = data.B.output.generation;
      const configA = {
        "llm" : data.A.llm,
        "retriever" : data.A.retriever,
        "rag" : data.A.rag,
      }
      const configB = {
        "llm" : data.B.llm,
        "retriever" : data.B.retriever,
        "rag" : data.B.rag,
      }
      storeObject("configA",configA);
      storeObject("configB",configB);

    })
    .catch(error => {
      console.error('Error fetching responses:', error);
      alert('Failed to get responses. Please try again.');
    })
    .finally(() => {
      hideSpinner();
    });
    ;
  }

  // Handle voting for a particular model
  function handleVote(model) {
    const configA = getObject("configA");
    const configB = getObject("configB");
    const winner = model === "A" ? configA : configB;
    const loser = model === "A" ? configB : configA;
    fetch(`${API_URL}/vote`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        winner: winner,
        loser: loser
      })
    })
    .then(response => response.json())
    .then(data => {
      alert(data.message); // Display the "투표가 완료되었습니다." message
      const messageBox = document.getElementById(`vote_${model.toLowerCase()}_message`);
      messageBox.value = `You voted for Model ${model}!`;
      messageBox.classList.remove("hidden");
    })
    .catch(error => {
      console.error('Error submitting vote:', error);
      alert('Failed to submit vote. Please try again.');
    });
  }
}
