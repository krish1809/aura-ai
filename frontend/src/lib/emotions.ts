import type { Emotion } from './types'

export const EMOTION_LABEL: Record<Emotion, string> = {
  anger: 'Anger',
  disgust: 'Disgust',
  fear: 'Anxiety / fear',
  joy: 'Joy',
  sadness: 'Sadness',
  surprise: 'Surprise',
  neutral: 'Calm / neutral',
}

export const emotionColor = (e: Emotion | string) => `var(--emo-${e})`

// Present-tense labels for the support strategy Aura chose for a reply.
export const STRATEGY_LABEL: Record<string, string> = {
  Question: 'Asked to understand',
  'Restatement or Paraphrasing': 'Reflected back what you said',
  'Reflection of feelings': 'Named your feelings',
  'Self-disclosure': 'Shared a relatable experience',
  'Affirmation and Reassurance': 'Offered reassurance',
  'Providing Suggestions': 'Suggested a next step',
  Information: 'Shared information',
  Others: 'Offered support',
}

export const STRATEGY_HELP: Record<string, string> = {
  Question: 'A gentle, open question to understand your situation better.',
  'Restatement or Paraphrasing': 'Putting what you said in other words so you feel heard.',
  'Reflection of feelings': 'Naming and validating the feeling behind your words.',
  'Self-disclosure': 'A short, relatable experience so you feel less alone.',
  'Affirmation and Reassurance': 'Recognising your strengths and offering reassurance.',
  'Providing Suggestions': 'One practical, concrete idea you could try.',
  Information: 'Useful, factual information related to your situation.',
  Others: 'Warm, general support.',
}

export const TOPIC_LABEL: Record<string, string> = {
  'ongoing depression': 'Ongoing low mood',
  'job crisis': 'Work & career',
  'breakup with partner': 'Relationship / breakup',
  'problems with friends': 'Friendships',
  'academic pressure': 'Studies & exams',
  other: 'Something else',
}

export const pct = (x: number | undefined | null) => `${Math.round((x ?? 0) * 100)}%`
