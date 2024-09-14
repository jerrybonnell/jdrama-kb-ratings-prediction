import java.util.*;
public class Matching {
  //---- constants
  public static final int KILL = -2;
  public static final int UNKNOWN = -1;
  public static final int ORIGINAL = 0;
  public static final int NOSPACE = 1;
  public static final int REVISED = 2;
  public static final int PREFIX = 3;
  public static final int DP = 4;
  public static final int MANUAL = 5;
  private static final String[] MATCHTYPENAMES = {
      "ORIGINAL", "NOSPACE", "REVISED", "PREFIX", "DP", "MANUAL" };
  public static final double MIN = 0.50;
  public static final String HEADER = "\"MATCHED\",\"TYPE\",\"DISTANCE\"";


  //---- corrections information
  private static String[] CORRECTIONS = {
    "ウォーターボーイズ", "WATER BOYS",        // "ビーチボーイズ (テレビドラマ)"
    "ウォーターボーイズ2", "WATER BOYS",        // "ビーチボーイズ (テレビドラマ)"
    "離婚弁護士II　〜ハンサムウーマン〜", "離婚弁護士II　〜ハンサムウーマン〜",        // "ハンサムマン"
    "ライアーゲーム シーズン2", "LIAR GAME (テレビドラマ)",        // "ゲームシェイカーズ"
    "MOZU　Season2 〜幻の翼〜", "MOZU",        // "猫侍 SEASON2"
    "RESET", "リセット (漫画)",        // "Silent (テレビドラマ)"
    "SUITS　season2", "SUITS/スーツ",        // "猫侍 SEASON2"
    "LOVE GAME", "LOVE&amp;PEACE (テレビドラマ)",
    "リアル・クローズ", "Real Clothes",        // "グリーンローズ"
    "美丘　−君がいた日々−", "美丘",        // "H2〜君といた日々"
    "ハガネの女 season2", "ハガネの女",        // "猫侍 SEASON2"
    "LAST HOPE　ラストホープ", "ラストホープ",        // "THE LAST COP/ラストコップ"
    "ファイアーボーイズ　〜め組の大吾〜", "FIRE BOYS 〜め組の大吾〜",        // ""
    "87％　−私の5年生存率−", "87%",        // ""
    "毒 ", "毒 ポイズン",        // ""
    "医龍2", "医龍-Team Medical Dragon- (テレビドラマ)",        // ""
    "光とともに･･･　〜自閉症児を抱えて〜", "光とともに…",        // ""
    "大奥　〜誕生〜[有功・家光篇]", "大奥〜誕生［有功・家光篇］",
    "大奥 〜第一章〜", "大奥 (フジテレビの時代劇)",        // "大奥 (2023年のテレビドラマ)"
    "大奥 〜華の乱〜", "大奥 (フジテレビの時代劇)",        // "大奥 (2023年のテレビドラマ)"
    "あしたの、喜多善男　〜世界一不運な男の、奇跡の11日間〜", "KILL", // ""
    "トップキャスター", "KILL"        // "トップリーグ (小説)"
  };

  //---- instance variables
  protected int qNum,mNum;
  protected String[] originalQ, originalM;
  protected String[] nospaceQ, nospaceM;
  protected String[] revisedQ, revisedM;
  protected int[] mapping;
  protected int[] matchType;
  protected double[] distance;
  protected boolean[] available;
  protected int[] matchCounts = new int[ MATCHTYPENAMES.length ];

  /**
   * constructor
   * @param	q	string array no.1
   * @param	m	string array no.2
   */
  Matching( String[] q, String[] m ) {
    // generate three lists from q and m
    int k;
    String w;
    qNum = q.length;
    originalQ = new String[qNum];
    nospaceQ = new String[qNum];
    revisedQ = new String[qNum];
    mapping = new int[qNum];
    matchType = new int[qNum];
    distance = new double[qNum];
    for ( int i = 0; i < qNum; i ++ ) {
      w = q[i];
      originalQ[i] = w;
      w = q[i].replace( " ", "").replace( "　", "" );
      nospaceQ[i] = w;
      w = revise( w );
      revisedQ[i] = w;
      mapping[i] = UNKNOWN;
      matchType[i] = UNKNOWN;
      distance[i] = 0;
    }
    mNum = m.length;
    originalM = new String[mNum];
    nospaceM = new String[mNum];
    revisedM = new String[mNum];
    available = new boolean[mNum];
    for ( int i = 0; i < mNum; i ++ ) {
      w = m[i];
      originalM[i] = w;
      w = w.replace( " ", "").replace( "　", "" );
      nospaceM[i] = w;
      w = revise( w );
      revisedM[i] = w;
      available[i] = true;
    }
  }

  /**
   * revise a name string with replacements
   * @param	w	the input
   * @return	the string after revision
   */
  public static String revise( String w ) {
    int k;
    w = w.toUpperCase();
    w = w.replace( "〜", "" ).replace( "？", "?" ).replace( "！", "!")
         .replace( "☆", "*" ).replace( "★", "*" ).replace( "＃", "#" )
         .replace( "／", "" ).replace( "/", "" ).replace( "−", "" )
         .replace( "-", "" ).replace( "・", "" ).replace( "、", "" )
         .replace( "&AMP;", "&" ).replace( "＆", "&" );
    k = w.indexOf( "（" );
    if ( k < 0 ) k = w.indexOf( "(" );
    return ( k > 0 ) ? w.substring( 0, k ) : w;
  }

  /**
   * record a match
   * @param	j	the index on q
   * @param	k	the index on m
   * @param	type	the match type
   */
  void addInfo( int j, int k, int type ) {
    System.out.printf( "%s:%s:%s:%s%n",
        originalQ[j], originalM[k],
        revisedQ[j], revisedM[k] );
    mapping[j] = k;
    matchType[j] = type;
    matchCounts[type] ++;
    available[k] = false;
  }

  /**
   * compare originals
   */
  void matchOriginal() {
    System.out.println( "------ COMPARE ORIGINALS -----" );
    for ( int j = 0; j < qNum; j ++ ) {
      if ( mapping[j] == UNKNOWN ) {
        for ( int k = 0; k < mNum; k ++ ) {
          if ( available[k] && originalQ[j].equals( originalM[k] ) ) {
            addInfo( j, k, ORIGINAL );
            break;
          }
        }
      }
    }
  }

  /**
   * compare strings without space
   */
  void matchNospace() {
    System.out.println( "------ COMPARE WITHOUT SPACE -----" );
    for ( int j = 0; j < qNum; j ++ ) {
      if ( mapping[j] == UNKNOWN ) {
        for ( int k = 0; k < mNum; k ++ ) {
          if ( available[k] && nospaceQ[j].equals( nospaceM[k] ) ) {
            addInfo( j, k, NOSPACE );
            break;
          }
        }
      }
    }
  }

  /**
   * compare strings after revision
   */
  void matchRevised() {
    System.out.println( "------ COMPARE REVISED -----" );
    for ( int j = 0; j < qNum; j ++ ) {
      if ( mapping[j] == UNKNOWN ) {
        for ( int k = 0; k < mNum; k ++ ) {
          if ( available[k] && revisedQ[j].equals( revisedM[k] ) ) {
            addInfo( j, k, REVISED );
            break;
          }
        }
      }
    }
  }

  /**
   * prefix matching
   */
  void matchPrefix() {
    System.out.println( "------ COMPARE PREFIX -----" );
    for ( int j = 0; j < qNum; j ++ ) {
      if ( mapping[j] == UNKNOWN ) {
        for ( int k = 0; k < mNum; k ++ ) {
          if ( available[k] == true &&
              ( revisedM[k].startsWith( revisedQ[j] ) &&
                revisedQ[j].length() >= 3 ||
                revisedQ[j].startsWith( revisedM[k] ) &&
                revisedM[k].length() >= 3 ) ) {
            addInfo( j, k, PREFIX );
            break;
          }
        }
      }
    }
  }

  /**
   * matching with corrections
   */
  void matchManual() {
    System.out.println( "------ COMPARE MANUAL ENTRIES -----" );
    int ell;
    String w;
    for ( int j = 0; j < qNum; j ++ ) {
      if ( mapping[j] == UNKNOWN ) {
        ell = 0;
        while ( ell < CORRECTIONS.length &&
            !originalQ[j].equals( CORRECTIONS[ell] ) ) ell += 2;
        if ( ell < CORRECTIONS.length ) {
          w = CORRECTIONS[ell+1];
          if ( w.equals( "KILL" ) ) mapping[j] = KILL;
          else {
            for ( int k = 0; k < mNum; k ++ ) {
              if ( w.equals( originalM[k] ) ) {
                addInfo( j, k, MANUAL );
                break;
              }
            }
          }
        }
      }
    }
  }

  /**
   * matching using dynamic programming
   */
  void matchDP() {
    System.out.println( "------ DYNAMIC PROGRAMMING -----" );
    double d, maxSim = MIN;
    double[] sim;
    int position, index1, index2;
    for ( int j = 0; j < qNum; j ++ ) {
      maxSim = MIN;
      position = UNKNOWN;
      if ( mapping[j] == UNKNOWN ) {
        for ( int k = 0; k < mNum; k ++ ) {
          if ( true ) { // available[k] == true ) {
            sim = DPDistance.similarityRatio( revisedQ[j], revisedM[k] );
            d = sim[DPDistance.AVE_POS];
            if ( d >= maxSim ) {
              maxSim = d;
              position = k;
            }
          }
        }
        if ( position >= 0 ) {
          addInfo( j, position, DP );
          distance[j] = maxSim;
        }
      }
    }
  }

  /**
   * execute all the matches and report
   */
  void match() {
    matchOriginal();
    matchNospace();
    matchRevised();
    matchPrefix();
    matchManual();
    matchDP();
    for ( int i = 0; i < matchCounts.length; i ++ ) {
      System.out.printf( "%s:%d%n",
          MATCHTYPENAMES[i], matchCounts[i] );
    }
  }

  /**
   * @return	the mapping
   * @param	qIndex	the input index
   */
  int getMapping( int qIndex ) {
    return mapping[qIndex];
  }

  /**
   * @return	signature of the matching
   * @param	qIndex	the input index
   */
  String matchSignature( int qIndex ) {
     int k = mapping[qIndex];
     int l = matchType[qIndex];
     double d = distance[qIndex];
     String matched = ( k < 0 ) ? "UNKNOWN" : originalM[k];
     String type = ( l < 0 ) ? "UNKNOWN" : MATCHTYPENAMES[l];
     return String.format( "\"%s\",\"%s\",\"%.3f\"",
         matched, type, d );
  }
}
