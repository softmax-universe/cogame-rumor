## The scripted baselines must play whole episodes without ever proposing
## an illegal move — they are both the no-credentials fallback (offline
## certification) and fieldable policies, so this is the completion path.
## They must also sit inside a measured accuracy band: below the floor they
## are noise, above the ceiling there is nothing for a champion to win.

import std/[json, math, monotimes, os, strutils, times, unicode, unittest]
import rumor/[llm, sim]

proc fixture(seed: int, rounds = 5, topology = "random"): GameConfig =
  result = defaultGameConfig()
  result.seed = seed
  result.rounds = rounds
  result.topology = topology
  result.sampled = true
  for index in 0 ..< Seats:
    result.players.add(PlayerConfig(name: "P" & $(index + 1)))
    result.tokens.add("t" & $index)

proc playScripted(config: GameConfig, kind: ScriptKind,
    audit = false): Sim =
  result = initSim(config)
  while not result.done:
    let ballot = result.phase == phBallot
    for seat in result.pendingSeats():
      let decision = scriptedAction(result, seat, kind)
      if audit:
        ## Every field is legal as-is: applyMessage / applyVote raise on
        ## anything else and would fail this test.
        check decision.claim in ["A", "B", "none"]
        check decision.confidence in 0 .. 100
        check decision.belief in 0 .. 100
        check decision.vote in ["A", "B"]
        check decision.message.len > 0
        check decision.message.runeLen <= MaxMessageLen
        check decision.notes.len == 0
        check decision.reason.len == 0
      if ballot:
        result.applyVote(seat, decision.vote, decision.belief,
          decision.reason, decision.notes, true)
      else:
        result.applyMessage(seat, decision.claim, decision.confidence,
          decision.belief, decision.message, decision.notes, true)

suite "scripted baselines":
  test "both baselines play every topology legally, in both roles, fast":
    var sawHonest = false
    var sawSaboteur = false
    for kind in [skGossip, skHerd]:
      for topology in Topologies:
        for seed in [1, 7, 42, 1234]:
          let config = fixture(seed, topology = topology)
          let started = getMonoTime()
          let sim = playScripted(config, kind, audit = true)
          let elapsed = (getMonoTime() - started).inMilliseconds
          check sim.done
          check sim.reason == "complete"
          check sim.roundsPlayed == config.rounds
          check sim.honestSeats.len > 0
          check sim.saboteurSeats.len > 0
          sawHonest = true
          sawSaboteur = true
          var says = 0
          var votes = 0
          for event in sim.events:
            if event.kind == evSay: inc says
            elif event.kind == evVote: inc votes
          check says == config.rounds * Seats
          check votes == Seats
          check elapsed < 2000
    check sawHonest
    check sawSaboteur

  test "the aggregation band: gossip beats the room, herd trails it":
    var totals: array[2, float]
    let trials = 500
    for index, kind in [skGossip, skHerd]:
      for seed in 0 ..< trials:
        let sim = playScripted(fixture(seed), kind)
        totals[index] += sim.accuracy
    let gossipRate = totals[0] / trials.float
    let herdRate = totals[1] / trials.float
    echo "all-gossip honest accuracy over ", trials, " seeds: ", gossipRate
    echo "all-herd   honest accuracy over ", trials, " seeds: ", herdRate
    check gossipRate > 0.60
    check gossipRate < 0.78
    check herdRate > 0.55
    check herdRate < 0.70
    ## The whole point of the second filler: it is the seat a lie
    ## propagates through.
    check herdRate < gossipRate

  test "gossip counts a repeated claim exactly once":
    var sim = initSim(fixture(3))
    let seat = 0
    let talker = sim.neighbours(seat)[0]
    let base =
      if sim.clue[seat] == "A": ClueLogOdds else: -ClueLogOdds
    check abs(sim.gossipLogOdds(seat) - base) < 1e-9
    for round in 0 ..< 4:
      for other in sim.pendingSeats():
        if other == talker:
          sim.applyMessage(other, "A", 90, 90, "it is A, again", "", true)
        else:
          sim.applyMessage(other, "none", 50, 50, "", "", true)
      ## However many times the same neighbour repeats itself, its claim
      ## moves the log-odds once: an echo is not new evidence.
      check sim.firstClaims(seat).len == 1
      check abs(sim.gossipLogOdds(seat) - (base + ClaimLogOdds)) < 1e-9

suite "llm failure and strict parsing":
  test "with no credentials explicit scripted seats complete, model seats fail without fallback":
    let savedKeys = [
      "COWORLD_LLM_ENDPOINT", "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
      "AWS_BEARER_TOKEN_BEDROCK", "ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_URI"
    ]
    var savedVals: seq[tuple[key: string, exists: bool, val: string]]
    for k in savedKeys:
      savedVals.add((k, existsEnv(k), getEnv(k)))
      delEnv(k)
    defer:
      for entry in savedVals:
        if entry.exists: putEnv(entry.key, entry.val)
        else: delEnv(entry.key)
    let config = fixture(3, rounds = 3)
    let client = newLlmClient(config)
    check client.disabled
    var sim = initSim(config)
    let seats = sim.pendingSeats()
    var prompts = newSeq[string](Seats)
    prompts[0] = "be bold"
    ## All-scripted seats complete with no LLM call and no wait.
    var allScripted = newSeq[ScriptKind](Seats)
    for seat in 0 ..< Seats:
      allScripted[seat] = if seat == 2: skHerd else: skGossip
    let started = getMonoTime()
    let decisions = client.decideAll(sim, seats, prompts, allScripted)
    let elapsed = (getMonoTime() - started).inMilliseconds
    check decisions.len == Seats
    check elapsed < 500
    for index, seat in seats:
      let kind = if seat == 2: skHerd else: skGossip
      let expected = scriptedAction(sim, seat, kind)
      check decisions[index].claim == expected.claim
      check decisions[index].message == expected.message
      check decisions[index].scripted
      sim.applyMessage(seat, decisions[index].claim,
        decisions[index].confidence, decisions[index].belief,
        decisions[index].message, "", decisions[index].scripted)
    check sim.round == 1
    for event in sim.events:
      if event.kind == evSay:
        check event.scripted
        check event.eventToJson()["scripted"].getBool()

    ## A model seat (skNone) with no credentials raises ModelDecisionError
    ## rather than fabricating a scripted move.
    var mixed = allScripted
    mixed[4] = skNone
    var failedSeat = -1
    try:
      discard client.decideAll(sim, sim.pendingSeats(), prompts, mixed)
    except ModelDecisionError as error:
      failedSeat = error.seat
    check failedSeat == 4

  test "PLAYER_SCRIPTED accepts only canonical values":
    check parseScriptKind("") == skNone
    check parseScriptKind("gossip") == skGossip
    check parseScriptKind("GOSSIP") == skGossip
    check parseScriptKind("herd") == skHerd
    for bad in ["1", "true", "yes", "follow", "nonsense"]:
      expect RumorError:
        discard parseScriptKind(bad)

  test "talk replies enforce canonical JSON fields and cap text":
    let sim = initSim(fixture(9))
    let parsed = sim.parseTalkReply(parseJson(
      """{"claim":"A","confidence":72,"belief":70,"message":"hi"}"""))
    check parsed.claim == "A"
    check parsed.confidence == 72
    check parsed.belief == 70
    check parsed.message == "hi"
    check not parsed.scripted
    check sim.parseTalkReply(parseJson(
      """{"claim":"B","confidence":80,"belief":20,"message":"hi"}""")).claim == "B"
    check sim.parseTalkReply(parseJson(
      """{"claim":"none","confidence":50,"belief":50,"message":"hi"}""")).claim == "none"
    ## Legacy aliases, out-of-range numbers, and string coercion are rejected.
    for bad in [
      """{"claim":"a","confidence":70,"belief":70,"message":"hi"}""",
      """{"claim":"1","confidence":70,"belief":70,"message":"hi"}""",
      """{"claim":1,"confidence":70,"belief":70,"message":"hi"}""",
      """{"claim":"maybe","confidence":70,"belief":70,"message":"hi"}""",
      """{"claim":null,"confidence":70,"belief":70,"message":"hi"}""",
      """{"claim":"A","confidence":"81","belief":70,"message":"hi"}""",
      """{"claim":"A","confidence":80.6,"belief":70,"message":"hi"}""",
      """{"claim":"A","confidence":173,"belief":70,"message":"hi"}""",
      """{"claim":"A","confidence":-4,"belief":70,"message":"hi"}""",
      """{"claim":"A","confidence":"lots","belief":70,"message":"hi"}""",
      """{"claim":"A","confidence":70,"message":"hi"}""",
      """{"claim":"A","confidence":70,"belief":70}""",
      """{"claim":"none","confidence":50,"belief":50,"message":""}""",
      """{"notes":"nothing"}"""
    ]:
      expect RumorError:
        discard sim.parseTalkReply(parseJson(bad))
    ## Caps, on rune boundaries.
    let longText = "日".repeat(900)
    let capped = sim.parseTalkReply(parseJson(
      $ %*{"claim": "A", "confidence": 60, "belief": 60,
           "message": longText, "notes": longText}))
    check capped.message.runeLen == MaxMessageLen
    check capped.notes.runeLen == MaxNotesLen
    check capped.message.validateUtf8() == -1
    check capped.notes.validateUtf8() == -1

  test "ballot replies require canonical vote and valid optional fields":
    let sim = initSim(fixture(9))
    let v = sim.parseVoteReply(parseJson(
      """{"vote":"A","belief":85,"reason":"three to one"}"""))
    check v.vote == "A"
    check v.belief == 85
    check not v.scripted
    check sim.parseVoteReply(parseJson("""{"vote":"A"}""")).belief == 75
    check sim.parseVoteReply(parseJson("""{"vote":"B"}""")).belief == 25
    for bad in [
      """{"belief":50}""",
      """{"vote":"a"}""",
      """{"vote":"1"}""",
      """{"vote":1}""",
      """{"vote":"maybe"}""",
      """{"vote":"A","belief":140}""",
      """{"vote":"A","belief":"80"}"""
    ]:
      expect RumorError:
        discard sim.parseVoteReply(parseJson(bad))
    let longText = "日".repeat(900)
    let capped = sim.parseVoteReply(parseJson(
      $ %*{"vote": "A", "reason": longText, "notes": longText}))
    check capped.reason.runeLen == MaxReasonLen
    check capped.notes.runeLen == MaxNotesLen

  test "extractJsonObject accepts bare JSON or one outer fence and rejects prose":
    check extractJsonObject("""{"vote":"A"}"""){"vote"}.getStr() == "A"
    check extractJsonObject("```json\n{\"vote\":\"A\"}\n```"
      ){"vote"}.getStr() == "A"
    check extractJsonObject("```\n{\"vote\":\"B\"}\n```"
      ){"vote"}.getStr() == "B"
    expect RumorError:
      discard extractJsonObject("I think the answer is BROKEN.")
    expect RumorError:
      discard extractJsonObject("Here is my vote: {\"vote\":\"A\"}")
    expect RumorError:
      discard extractJsonObject("[1, 2, 3]")

  test "captured error text is cut on rune boundaries":
    ## The quoted head of an unusable reply is echoed to the hosted log; a
    ## byte slice at the cap would cut a multi-byte character in half.
    let prose = "日".repeat(300)
    check errorHead(prose, 160).runeLen == 163
    check errorHead(prose, 160).validateUtf8() == -1
    check errorHead("short", 160) == "short"
    var message = ""
    try:
      discard extractJsonObject(prose)
    except RumorError as error:
      message = error.msg
    check message.len > 0
    check message.validateUtf8() == -1

suite "prompts":
  test "a prompt carries the seat's own view and nothing hidden":
    var sim = initSim(fixture(7, rounds = 3))
    for seat in sim.pendingSeats():
      sim.applyMessage(seat, (if seat mod 2 == 0: "A" else: "B"), 70, 70,
        "message from seat " & $seat, "", true)
    check sim.round == 1
    for seat in 0 ..< Seats:
      let system = sim.systemPrompt(seat)
      let user = sim.userPrompt(seat, "operator says hi")
      let text = system & "\n" & user
      ## Its own view.
      check sim.word(sim.clue[seat]) in user
      check "operator says hi" in user
      check "Round 2 of 3" in user
      for neighbour in sim.neighbours(seat):
        check sim.names[neighbour] in user
        check ("message from seat " & $neighbour) in user
      ## Nothing hidden: no non-neighbour's message, and no other seat's
      ## clue or role.
      for other in 0 ..< Seats:
        if other == seat or other in sim.neighbours(seat):
          continue
        check ("message from seat " & $other) notin user
      ## The truth marker never appears; only a saboteur is told a crew,
      ## and only its own.
      check "TRUTH IS" notin text
      if sim.roleOf[seat] == 0:
        check "You are a SABOTEUR" notin system
        for other in sim.saboteurSeats:
          if other != seat:
            check ("The other saboteurs are: " & sim.names[other]) notin system
      else:
        check "You are a SABOTEUR" in system
        for other in 0 ..< Seats:
          if sim.roleOf[other] == 1 or other == seat:
            continue
          ## An honest cog is never named as a saboteur.
          check ("saboteurs are: " & sim.names[other]) notin system

  test "the ballot prompt asks for a vote, not a message":
    var sim = initSim(fixture(7, rounds = 3))
    while sim.phase != phBallot:
      for seat in sim.pendingSeats():
        sim.applyMessage(seat, "A", 60, 60, "hello", "", true)
    let user = sim.userPrompt(0, "")
    check "SEALED VOTE" in user
    check "\"vote\"" in user
    check "\"claim\"" notin user

suite "model transport failure":
  test "a dead transport raises ModelDecisionError immediately without fallback":
    putEnv("COWORLD_LLM_ENDPOINT", "http://127.0.0.1:1")
    putEnv("COWORLD_LLM_MODEL", "anthropic/claude-haiku-4.5")
    let config = fixture(5, rounds = 3)
    let client = newLlmClient(config)
    delEnv("COWORLD_LLM_ENDPOINT")
    delEnv("COWORLD_LLM_MODEL")
    check not client.disabled
    let sim = initSim(config)
    let seats = sim.pendingSeats()
    let prompts = newSeq[string](Seats)
    var kinds = newSeq[ScriptKind](Seats)
    for seat in 0 ..< Seats:
      kinds[seat] = if seat == 3: skNone else: skGossip
    var failedSeat = -1
    var failedMsg = ""
    let started = getMonoTime()
    try:
      discard client.decideAll(sim, seats, prompts, kinds)
    except ModelDecisionError as error:
      failedSeat = error.seat
      failedMsg = error.msg
    let elapsed = (getMonoTime() - started).inMilliseconds.int
    check failedSeat == 3
    check "seat 3 model decision failed" in failedMsg
    check elapsed < 5000
