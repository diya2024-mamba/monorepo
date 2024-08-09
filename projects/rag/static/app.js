document.addEventListener('DOMContentLoaded', function() {
    const msgerForm = get(".msger-inputarea");
    const msgerInput = get(".msger-input");
    const msgerChat = get(".msger-chat");
    const botNameSelect = get("#bot-name-select");

    let BOT_NAME;
    const USE_API = false; // Set this to true when API is ready

    const BOT_MSGS = [
      "Hi, how are you?",
      "Ohh... I can't understand what you trying to say. Sorry!",
      "I like to play games... But I don't know how to play!",
      "Sorry if my answers are not relevant. :))",
      "I feel sleepy! :("
    ];

    const PERSON_IMG = "./static/images/user.png";
    const PERSON_NAME = "User";

    // Fetch bot names from the JSON file
    fetch('./static/botNames.json')
      .then(response => response.json())
      .then(data => {
        const botNames = data.botNames;
        botNames.forEach(name => {
          const option = document.createElement('option');
          option.value = name;
          option.textContent = name;
          botNameSelect.appendChild(option);
        });
        BOT_NAME = botNameSelect.value;
      })
      .catch(error => console.error('Error fetching bot names:', error));

    botNameSelect.addEventListener('change', function() {
      BOT_NAME = this.value;
    });

    msgerForm.addEventListener("submit", event => {
      event.preventDefault();

      const msgText = msgerInput.value;
      if (!msgText) return;

      appendMessage(PERSON_NAME, PERSON_IMG, "right", msgText);
      msgerInput.value = "";

      handleBotResponse(msgText, BOT_NAME);
    });

    function appendMessage(name, img, side, text) {
      // Simple solution for small apps
      const msgHTML = `
        <div class="msg ${side}-msg">
          <div class="msg-img" style="background-image: url(${img})"></div>

          <div class="msg-bubble">
            <div class="msg-info">
              <div class="msg-info-name">${name}</div>
              <div class="msg-info-time">${formatDate(new Date())}</div>
            </div>

            <div class="msg-text">${text}</div>
          </div>
        </div>
      `;

      msgerChat.insertAdjacentHTML("beforeend", msgHTML);
      msgerChat.scrollTop += 500;
    }

    function handleBotResponse(userMessage, botName) {
      const BOT_IMG = "./static/images/bot_profile/" + botName + ".png"
      if (USE_API) {
        fetchBotResponseFromAPI(userMessage, botName).then(botMessage => {
          appendMessage(botName, BOT_IMG, "left", botMessage);
        });
      } else {
        const botMessage = getLocalBotResponse(botName);
        appendMessage(botName, BOT_IMG, "left", botMessage);
      }
    }

    function getLocalBotResponse(botName) {
      const r = random(0, BOT_MSGS.length - 1);
      return `I am ${botName}. ${BOT_MSGS[r]}`;
    }

    async function fetchBotResponseFromAPI(userMessage, botName) {
      try {
        const response = await fetch('http://your-api-url/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ message: userMessage, botName: botName })
        });
        const data = await response.json();
        return data.reply;
      } catch (error) {
        console.error('Error fetching bot response:', error);
        return "Sorry, I couldn't connect to the server.";
      }
    }

    // Utils
    function get(selector, root = document) {
      return root.querySelector(selector);
    }

    function formatDate(date) {
      const h = "0" + date.getHours();
      const m = "0" + date.getMinutes();

      return `${h.slice(-2)}:${m.slice(-2)}`;
    }

    function random(min, max) {
      return Math.floor(Math.random() * (max - min) + min);
    }
  });
