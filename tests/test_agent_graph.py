from backend.models.agent_graph import AgenticGraph

agent = AgenticGraph()

tests = [
    "I am feeling under the weather",
    "I have fever and headache",
    "What is Chicken Pox?",
]

for i in range(len(tests)):
    print("\n===== Query:", tests[i])
    output = agent.run(user_query=tests[i],session_id=str(i))
    print("Final Answer:", output["final_answer"]["rag_answer"])
    print("Mode:", output["final_answer"]["mode"])
    print("Symptoms:", output.get("symptoms"))
